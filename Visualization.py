#!/usr/bin/env python
# coding: utf-8
# Author: Arindam Majee
# Email: MAJEEARINDAM06072002[AT][GMAIL][DOT][COM]


# Import libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import zoom

import os
import sys
import json
import time
import datetime
from HCCT import *

print("All Import Done......")

# Clear Torch Cache
torch.cuda.empty_cache()
torch.manual_seed(0)
print("Cleared Torch Cache..............")\

# Input Data Preparation
if len(sys.argv) != 2:
    print("Invalid Arguments provided. Please run the script like - python Visualization.py {data-folder}")
    exit()

folder_path = sys.argv[1]
processed_data_folder = f"{folder_path}-processed"
os.makedirs(f"{processed_data_folder}", exist_ok=True)
os.system(f"nppy -i {folder_path} -o {processed_data_folder}/ -s -g -w -1")


# Config Setup
with open('config.json', 'r') as f:
    config = json.load(f)
model_cpfile = config['chkp-file']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
save_dir = f"{config['exp_name']}_Output"
os.makedirs(save_dir, exist_ok=True)

print("Data Preparation and output folder creation done!.........")


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def load_model(model_cp_path, model_config):
    model = ViTForClassfication(model_config).to(device)
    model.load_state_dict(torch.load(model_cp_path))
    return model


def get_last_layer_activations_grds(model, x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    if x.dim()==4:
        x = torch.unsqueeze(x, dim=0)
    x = x.to(device)
    h = model.embedding.patch_embeddings.conv_5.maxpool.register_forward_hook(get_activation('last_layer'))
    output, attention_maps = model(x, output_attentions=True)
    h.remove()
    forward_features = activation['last_layer']
    #forward_features[forward_features < 0] = 0
    
    return output, forward_features, attention_maps


def get_class_attention_coefficients(attention_maps: list):
    attention_maps = torch.cat(attention_maps, dim=1)
    # select only the attention maps of the CLS token
    attention_coeff = attention_maps[:, :, 0, 1:] 
    attention_coeff = torch.sum(attention_coeff, dim=1)
    attention_coeff = torch.squeeze(attention_coeff)
    attention_coeff = attention_coeff/torch.max(attention_coeff)
    return attention_coeff


def get_heatmap(model, forward_attentions, attention_coeff, return_map=True):
    forward_attentions = torch.squeeze(forward_attentions)
    for i in range(512):
        #forward_attentions[i, :, :, :] *= pooled_gradients[i]
        forward_attentions[i, :, :, :] *= attention_coeff[i]
    heatmap = torch.mean(forward_attentions, dim=0)
    heatmap = heatmap.detach().cpu().numpy()
    heatmap = np.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0)
    upscaled_heatmap = zoom(heatmap, (32, 32, 32), mode='nearest')

    upscaled_heatmap = np.uint8(upscaled_heatmap*255)

    if return_map:
        return heatmap, upscaled_heatmap
    else:
        return upscaled_heatmap

def plot_class_attention_map(attention_coeff, tag='label', cmap='hot'):
    h, w = 32, 16
    assert attention_coeff.size(dim=0) == h*w

    class_attention_map = attention_coeff.view(h, w)
    class_attention_map = class_attention_map/torch.max(class_attention_map)
    class_attention_map = class_attention_map.detach().cpu().numpy()
    class_attention_map = np.uint8(class_attention_map*255)
    
    return class_attention_map


# In[7]:
image_path_list = []
for root, dirs, files in os.walk(processed_data_folder):
    for file_name in files:
        if "nii_norm.nii" in file_name:
            image_path_list.append(os.path.join(root, file_name))

print(f"Got total {len(image_path_list)} images.")

def read_nii_images(image_list):
    image_affines = []
    for image_filepath in image_list:
        original_image = nib.as_closest_canonical(nib.load(image_filepath))
        image = original_image.get_fdata()
        xdim, ydim, zdim = image.shape
        image = np.pad(image, [((256-xdim)//2, (256-xdim)//2), ((256-ydim)//2, (256-ydim)//2), ((256-zdim)//2, (256-zdim)//2)], 'constant', constant_values=0)
        #image = image.reshape(image.shape[2], image.shape[1], image.shape[0])

        width_factor = config['image_size'] / image.shape[0]
        height_factor = config['image_size'] / image.shape[1]
        depth_factor = config['image_size'] / image.shape[-1]

        image = zoom(image, (width_factor, height_factor, depth_factor), order=1)
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
        image = image.astype('float32')

        image_affines.append((image, original_image.affine))

    return image_affines


# Load Model
print("Loading the Model............\n")
model = load_model(model_cpfile, config)

print("\nModel Loaded. Starting to read processed images...............")
indices = list(range(len(image_path_list)))
image_affines = read_nii_images(image_path_list)
images = [image_affines[i][0] for i in indices]
print("Read all images.........")

heats = []
start_time = time.time()
for i in range(len(indices)):
    epoch_start_time = time.time()
    fig, axes = plt.subplots(1, 3, figsize=(4, 8))
    x = images[i]
    model.eval()
    yhat, forward_features, attention_maps = get_last_layer_activations_grds(model, x)
    attention_coeff = get_class_attention_coefficients(attention_maps)
    heatmap, upscaled_heatmap = get_heatmap(model, forward_features, attention_coeff)
    upscaled_heatmap = upscaled_heatmap * x[0]
    upscaled_heatmap = upscaled_heatmap/np.max(upscaled_heatmap)
    heats.append(heatmap)
    
    x = np.squeeze(x)
    slice_idx = x.shape[0]//2
    sagital = np.rot90(x[slice_idx, :, :])
    sagital_heatmap = np.rot90(upscaled_heatmap[slice_idx, :, :])
    coronal = np.rot90(x[:, slice_idx, :])
    coronal_heatmap = np.rot90(upscaled_heatmap[:, slice_idx, :])
    axial = np.rot90(x[:, :, slice_idx])
    axial_heatmap = np.rot90(upscaled_heatmap[:, :, slice_idx])
    
    mask = np.concatenate((np.ones((192, 192)), np.zeros((192, 192))), axis=1)
    sagital = np.concatenate((sagital, sagital), axis=1)
    sagital_heatmap = np.concatenate((sagital_heatmap, sagital_heatmap), axis=1)
    sagital_heatmap = np.ma.masked_where(mask==1, sagital_heatmap)
    coronal = np.concatenate((coronal, coronal), axis=1)
    coronal_heatmap = np.concatenate((coronal_heatmap, coronal_heatmap), axis=1)
    coronal_heatmap = np.ma.masked_where(mask==1, coronal_heatmap)
    axial = np.concatenate((axial, axial), axis=1)
    axial_heatmap = np.concatenate((axial_heatmap, axial_heatmap), axis=1)
    axial_heatmap = np.ma.masked_where(mask==1, axial_heatmap)
    
    
    axes[0].imshow(sagital, cmap='gray')
    axes[0].imshow(sagital_heatmap, alpha=1, cmap='jet')
    axes[1].imshow(coronal, cmap='gray')
    axes[1].imshow(coronal_heatmap, alpha=1, cmap='jet')
    axes[2].imshow(axial, cmap='gray')
    axes[2].imshow(axial_heatmap, alpha=1, cmap='jet')
    
    axes[0].axis(False)
    axes[1].axis(False)
    axes[2].axis(False)

    name = image_path_list[i].split("/")[-1].split(".")[0]
    for j in range(3):
        axes[0].set_title("sagital")
        axes[1].set_title("coronal")
        axes[2].set_title("axial")

    plt.savefig(os.path.join(save_dir, f'{name}-heatmap_vis.png'), dpi=900, bbox_inches='tight')

    new_image = nib.Nifti1Image(upscaled_heatmap, affine=image_affines[i][1])
    nib.save(new_image, os.path.join(save_dir, f'{name}-heatmap.nii.gz'))

    print(f"Finished visualization processing of {i+1} image. Took {time.time() - epoch_start_time} seconds. Total spend time {time.time() - start_time} seconds.")
