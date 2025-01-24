# -*- coding: utf-8 -*-
"""LLaVA_ov_attn_eval.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1QwjsorxdtNfyvRfmmZQLIDJ5yixIoy1Q
"""

# Commented out IPython magic to ensure Python compatibility.
#!pip install -U git+https://github.com/filippo-merlo/LLaVA-NeXT.git

import os
import sys
import gc
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import cv2
from PIL import Image
from pprint import pprint
import copy
from io import BytesIO
import requests
import json
import torch
import torch.nn.functional as F
import pandas as pd
import math

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from typing import Dict
from functools import partial, reduce
from PIL import Image
from transformers.image_processing_utils import get_size_dict
from transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import ChannelDimension, to_numpy_array, PILImageResampling
from transformers.utils import ModelOutput

# Utils for the task

def log_metrics(model_name, metrics):
    log_path = f"outputs/logs/{model_name}_log.json"
    with open(log_path, 'w') as f:
        json.dump(metrics, f)
    print(f"Logged metrics for {model_name}")


def add_grey_background_and_rescale_bbox(image_path, bbox):
    """
    Adds a grey square background to the input image, centers the image on it,
    and rescales the bounding box to correspond to the new image dimensions.

    Args:
        image_path (ster): Path to input image.
        bbox (tuple): A tuple describing the bounding box in the format (x, y, w, h).

    Returns:
        PIL.Image.Image: The new image with a grey background.
        tuple: The rescaled bounding box in the format (x, y, w, h).
    """
    image = Image.open(image_path)

    # Unpack the bounding box
    x, y, w, h = bbox

    # Determine the size of the new square background
    max_dim = max(image.width, image.height)
    new_size = (max_dim, max_dim)

    # Create a new grey background image
    grey_background = Image.new("RGB", new_size, color=(128, 128, 128))

    # Calculate the position to center the original image
    offset_x = (max_dim - image.width) // 2
    offset_y = (max_dim - image.height) // 2

    # Paste the original image onto the grey background
    grey_background.paste(image, (offset_x, offset_y))

    # Rescale the bounding box to the new image dimensions
    new_bbox = (x + offset_x, y + offset_y, w, h)

    return grey_background, new_bbox

def add_gaussian_noise_in_bbox(image, bbox, noise_level=0.0):

    # Add noise to the image within the bounding box
    image_np = np.array(image)

    # Box notation [x, y, w, h]
    x, y, w, h = map(int, bbox)

    # Ensure the bounding box is within the image dimensions
    x_end = min(x + w, image_np.shape[1])
    y_end = min(y + h, image_np.shape[0])

    # Extract the region of interest
    roi = image_np[y:y_end, x:x_end]

    # Generate Gaussian noise
    noise = np.random.normal(loc=0, scale=255*noise_level, size=roi.shape)

    # Add noise to the region of interest
    noisy_roi = roi + noise

    # Clip values to be valid pixel values
    noisy_roi = np.clip(noisy_roi, 0, 255).astype(np.uint8)

    # Replace the original region with the noisy one
    image_np[y:y_end, x:x_end] = noisy_roi

    # Convert back to PIL Image
    noisy_image = Image.fromarray(image_np)

    return noisy_image

def add_gaussian_noise_outside_bbox(image, bbox, noise_level=0.0):
    """
    Adds Gaussian noise to an image only outside the specified bounding box.

    Parameters:
        image (PIL.Image.Image): Input image.
        bbox (tuple): Bounding box in the format (x, y, w, h).
        noise_level (float): The standard deviation of the Gaussian noise as a fraction of the intensity range (0-1).

    Returns:
        PIL.Image.Image: Image with Gaussian noise applied outside the bounding box.
    """
    # Convert the image to a numpy array
    image_np = np.array(image)

    # Extract the bounding box coordinates
    x, y, w, h = map(int, bbox)
    x_end = min(x + w, image_np.shape[1])
    y_end = min(y + h, image_np.shape[0])

    # Create a mask for the bounding box region
    mask = np.zeros_like(image_np, dtype=bool)
    mask[y:y_end, x:x_end] = True

    # Generate Gaussian noise
    noise = np.random.normal(loc=0, scale=255 * noise_level, size=image_np.shape)

    # Apply noise only where the mask is False (outside the bbox)
    noisy_image_np = np.where(mask, image_np, image_np + noise)

    # Clip pixel values to ensure they are valid
    noisy_image_np = np.clip(noisy_image_np, 0, 255).astype(np.uint8)

    # Convert the numpy array back to a PIL Image
    noisy_image = Image.fromarray(noisy_image_np)

    return noisy_image

 # Normalize box diamentions

def normalize_box(bbox, image_width=1025, image_height=1025):
    return (
        round(float(bbox[0] / image_width), 4),
        round(float(bbox[1] / image_height), 4),
        round(float(bbox[2] / image_width), 4),
        round(float(bbox[3] / image_height), 4),
    )

def convert_box(bbox):
    x, y, w, h = tuple(bbox) # Box coordinates are in (left, top, width, height) format
    return [x, y, x+w, y+h]

# many are copied from https://github.com/mattneary/attention/blob/master/attention/attention.py
# here it nullifies the attention over the first token (<bos>)
# which in practice we find to be a good idea


def aggregate_llm_attention(attn):
    '''Extract average attention vector'''
    avged = []
    for layer in attn:
        layer_attns = layer.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        vec = torch.concat((
            # We zero the first entry because it's what's called
            # null attention (https://aclanthology.org/W19-4808.pdf)
            torch.tensor([0.]),
            # usually there's only one item in attns_per_head but
            # on the first generation, there's a row for each token
            # in the prompt as well, so take [-1]
            attns_per_head[-1][1:].cpu(),
            # attns_per_head[-1].cpu(),
            # add zero for the final generated token, which never
            # gets any attention
            torch.tensor([0.]),
        ))
        avged.append(vec / vec.sum())
    return torch.stack(avged).mean(dim=0)

def aggregate_vit_attention_subtract_avg(attn, attn_avg, select_layer=-2, all_prev_layers=True):
    """
    Highlights layer-specific attention by subtracting the average attention across all layers
    and ensures the results are non-negative (only positive values or 0).
    Parameters:
        attn: List of attention maps, one per layer. Each map is of shape (batch_size, num_heads, num_tokens, num_tokens).
        select_layer: Layer index to process (-2 by default, assuming LLaVA-style).
        all_prev_layers: If True, considers all layers up to `select_layer`; otherwise, processes only the specified layer.
    Returns:
        Adjusted attention maps for each layer after removing the average attention across all layers.
    """
    avged = []

    # Step 2: Subtract average attention from each layer and apply ReLU
    for i, layer in enumerate(attn):
        if all_prev_layers and i > len(attn) + select_layer:
            break
        layer_attns = layer.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        vec = attns_per_head[0:, 0:].cpu()
        vec_normalized = vec / vec.sum(-1, keepdim=True)
        # Subtract the average attention and apply ReLU to ensure non-negative values
        adjusted_attention = torch.relu(vec_normalized - attn_avg[i])
        avged.append(adjusted_attention)

    if all_prev_layers:
        return torch.stack(avged)  # Return all adjusted attention maps
    else:
        # For single layer mode, return adjusted attention for the selected layer
        selected_layer = attn[select_layer]
        layer_attns = selected_layer.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        vec = attns_per_head[0:, 0:].cpu()
        vec_normalized = vec / vec.sum(-1, keepdim=True)
        return torch.relu(vec_normalized - attn_avg[i])


def heterogenous_stack(vecs):
    '''Pad vectors with zeros then stack'''
    max_length = max(v.shape[0] for v in vecs)
    return torch.stack([
        torch.concat((v, torch.zeros(max_length - v.shape[0])))
        for v in vecs
    ])


def load_image(image_path_or_url):
    if image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
        response = requests.get(image_path_or_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_path_or_url).convert('RGB')
    return image


def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_HSV)
    hm = np.float32(heatmap) / 255
    cam = hm + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam), heatmap

# ===> specify the model path
pretrained = "lmms-lab/llava-onevision-qwen2-0.5b-si"
model_name = "llava_qwen"
device = "cuda"
cache_dir = '/mnt/cimec-storage6/shared/hf_lvlms'
device_map = "auto"

# load the model
load_8bit = False
load_4bit = False

llava_model_args = {
    "multimodal": True,
    "attn_implementation": "sdpa",
    "load_8bit" : load_8bit,
    "load_4bit" : load_4bit,
}

tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, **llava_model_args, cache_dir=cache_dir, device_map=device_map)
model.eval()

# Load Image And Load Data
def load_dataset(dataset_path):
    # Load and preprocess data
    with open(dataset_path, 'r') as f:
        return json.load(f)

def get_images_names_path(images_path):
    # Load and preprocess images
    images_n_p = {}
    for filename in os.listdir(images_path):
        if filename.endswith('.jpg'):
            images_n_p[filename] = os.path.join(images_path, filename)
    return images_n_p

images_path = "/mnt/cimec-storage6/users/filippo.merlo/sceneREG_data/resized_images"
dataset_path = "/mnt/cimec-storage6/users/filippo.merlo/sceneREG_data/final_dataset_resized.json"
# Specify the directory
output_dir = "/mnt/cimec-storage6/users/filippo.merlo/sceneREG_data/attention_deployment"
os.makedirs(output_dir, exist_ok=True)

# whole dataset avg visual attention matrix

avg_vis_attn_matrix_path = '/mnt/cimec-storage6/users/filippo.merlo/sceneREG_data/attention_deployment/vis_attn_matrix_average.pt'
avg_vis_attn_matrix = torch.load(avg_vis_attn_matrix_path)

data = load_dataset(dataset_path)
images_n_p = get_images_names_path(images_path)

noise_levels = [0.0, 0.5, 1.0]
conditions = ['target_noise','context_noise','all_noise']
evaluation_results = []

results_list = []

for condition in conditions:
  for noise_level in noise_levels:
    if conditions == 'context_noise' and noise_level == 0.0:
      continue
    elif conditions == 'all_noise' and noise_level == 0.0:
      continue

    for image_name, image_path  in tqdm(list(images_n_p.items())):
      if data[image_name]['excluded']:
        continue

      bbox = data[image_name]['target_bbox']

      if '_original.jpg' in image_name:
          target = data[image_name]['target'].replace('_', ' ')

      elif '_clean.jpg' in image_name:
          continue # skip clean images

      else:
          target = data[image_name]['swapped_object'].replace('_', ' ')

      original_image = load_image(image_path)
      original_image_size = original_image.size

      if original_image_size[0] != 640: ###!!!
          continue

      # get the image with a grey background and the bounding box rescaled
      image, bbox = add_grey_background_and_rescale_bbox(image_path, bbox)

      # get the image with the corresponding noise level in the roi
      if condition == 'target_noise':
          image = add_gaussian_noise_in_bbox(image, bbox, noise_level)
      elif condition == 'context_noise':
          image = add_gaussian_noise_outside_bbox(image, bbox, noise_level)
      elif condition == 'all_noise':
          image = add_gaussian_noise_in_bbox(image, bbox, noise_level)
          image = add_gaussian_noise_outside_bbox(image, bbox, noise_level)

      # Process input
      image_sizes = image.size
      W = image.size[0]
      H = image.size[1]
      normalized_bbox = normalize_box(convert_box(bbox), W, H)
      x1, y1, x2, y2 = normalized_bbox

      with torch.inference_mode():

        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor] ##

        conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
        prompt_text = DEFAULT_IMAGE_TOKEN + f"\nWhat is the object in this part of the image [{x1}, {y1}, {x2}, {y2}]? Answer with the object's name only. No extra text."
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], prompt_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

        # generate the response
        outputs = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_sizes],
            do_sample=False,
            temperature=0,
            max_new_tokens=10,
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=True,
        )

        del image_tensor

        text = tokenizer.decode(outputs["sequences"][0]).strip()

        # constructing the llm attention matrix
        aggregated_prompt_attention = []
        for i, layer in enumerate(outputs["attentions"][0]):
            layer_attns = layer.squeeze(0)
            del layer
            attns_per_head = layer_attns.mean(dim=0)
            del layer_attns
            cur = attns_per_head[:-1].cpu().clone()
            # following the practice in `aggregate_llm_attention`
            # we are zeroing out the attention to the first <bos> token
            # for the first row `cur[0]` (corresponding to the next token after <bos>), however,
            # we don't do this because <bos> is the only token that it can attend to
            cur[1:, 0] = 0.
            cur[1:] = cur[1:] / cur[1:].sum(-1, keepdim=True)
            aggregated_prompt_attention.append(cur)
        aggregated_prompt_attention = torch.stack(aggregated_prompt_attention).mean(dim=0)

        # llm_attn_matrix will be of torch.Size([N, N])
        # where N is the total number of input (both image and text ones) + output tokens
        llm_attn_matrix = heterogenous_stack(
            [torch.tensor([1])]
            + list(aggregated_prompt_attention)
            + list(map(aggregate_llm_attention, outputs["attentions"]))
        )

        del aggregated_prompt_attention

        # identify length or index of tokens
        input_token_len = model.get_vision_tower().num_patches + len(input_ids[0]) - 1 # -1 for the <image> token
        vision_token_start = len(tokenizer(prompt.split("<image>")[0], return_tensors='pt')["input_ids"][0])
        vision_token_end = vision_token_start + model.get_vision_tower().num_patches
        output_token_len = len(outputs["sequences"][0])
        output_token_start = input_token_len
        output_token_end = input_token_len + output_token_len

        del input_ids
        del outputs

        # connect with the vision encoder attention
        # to visualize the attention over the image

        # vis_attn_matrix will be of torch.Size([N, N])
        # where N is the number of vision tokens/patches
        # `all_prev_layers=True` will average attention from all layers until the selected layer
        # otherwise only the selected layer's attention will be used

        att_on_whole_image = []

        for v in model.get_vision_tower().image_attentions:
          att_on_whole_image.append(v[0])
          del v

        del model.get_vision_tower().image_attentions


        attn_over_target_list = []
        attn_over_target_max_list = []
        attn_over_context_list = []
        attn_over_context_max_list = []
        tot_attn_list = []
        attn_over_image_max_list = []
       

        for layer in list(range(0,26)):

            vis_attn_matrix = aggregate_vit_attention_subtract_avg(
                att_on_whole_image, ### att_on_whole_image
                attn_avg=avg_vis_attn_matrix,
                select_layer=layer,
                all_prev_layers=False
            )

            grid_size = model.get_vision_tower().num_patches_per_side ### grid_size

            # Define the range of output tokens, excluding the last one
            output_token_inds = list(range(output_token_start, output_token_end - 1))

            attn_over_image_final = None
            
            # Loop through all tokens except the last one
            for i, token_id in enumerate(output_token_inds):
                
                # Compute attention weights for vision tokens for the current token
                attn_weights_over_vis_tokens = llm_attn_matrix[token_id][vision_token_start:vision_token_end]
                attn_weights_over_vis_tokens = attn_weights_over_vis_tokens / attn_weights_over_vis_tokens.sum()

                attn_over_image = []
                
                # Calculate weighted attention maps over visual tokens
                for weight, vis_attn in zip(attn_weights_over_vis_tokens, vis_attn_matrix):
                    vis_attn = vis_attn.reshape(grid_size, grid_size)
                    attn_over_image.append(vis_attn * weight)
                
                # Combine the attention maps for this token
                attn_over_image = torch.stack(attn_over_image).sum(dim=0)
                attn_over_image = attn_over_image / attn_over_image.max()

                # Upsample to match the original image size
                attn_over_image = F.interpolate(
                    attn_over_image.unsqueeze(0).unsqueeze(0),
                    size=image.size, 
                    mode='nearest',
                ).squeeze()

                # Accumulate attention maps to average over tokens
                if attn_over_image_final is None:
                    attn_over_image_final = attn_over_image
                else:
                    attn_over_image_final += attn_over_image

            # Compute the average over all tokens except the last one
            attn_over_image_final /= len(output_token_inds)

            # METRICS

            attn_over_image = attn_over_image_final.to(torch.float64)

            x_min, y_min, w, h = bbox ### bbox

            target_area = w * h
            original_image_w = original_image_size[0] ### original_image_size
            original_image_h = original_image_size[1]
            grey_board_lenght = int((original_image_w - original_image_h)/2)
            tot_area = original_image_w * original_image_h

            context_area = tot_area - target_area

            x_max = x_min + w
            y_max = y_min + h

            y_min = int(y_min)
            y_max = int(y_max)
            x_min = int(x_min)
            x_max = int(x_max)

            attn_over_target = attn_over_image[y_min:y_max, x_min:x_max]
            attn_over_target_max = attn_over_target.max()
            attn_over_target = attn_over_target.sum().item()

            # Create a mask for the entire image (1 for context, 0 for target)
            mask = torch.ones_like(attn_over_image)
            mask[0:grey_board_lenght, :] = 0 # Upper board
            mask[original_image_w-grey_board_lenght:, :] = 0 # Lower board
            mask[y_min:y_max, x_min:x_max] = 0  # Set target area to 0

            # Compute attention over the context area
            attn_over_context = attn_over_image * mask
            attn_over_context_max = attn_over_context.max()
            attn_over_context = attn_over_context.sum().item()

            tot_attn = attn_over_image.sum().item()
            attn_over_image_max = attn_over_image.max()
            
            attn_over_target_list.append(attn_over_target)
            attn_over_target_max_list.append(attn_over_target_max)
            attn_over_context_list.append(attn_over_context)
            attn_over_context_max_list.append(attn_over_context)
            tot_attn_list.append(tot_attn)
            attn_over_image_max_list.append(attn_over_image_max)
            
            
            del attn_over_image
            del attn_over_image_final
            del mask
            del attn_weights_over_vis_tokens

            # Cleanup after each layer
            del vis_attn_matrix
          

        # Append the results as a dictionary to the list
        results_list.append({
            "image_name": image_name,
            "condition": condition,
            "noise_level": noise_level,
            "target": target,
            "bbox": bbox,
            "output_text": text,
            "grid_size": model.get_vision_tower().num_patches_per_side,
            "image_size": image.size,
            "original_image_size": original_image_size,
            'scene': data[image_name]['scene'],
            'rel_score': data[image_name]['rel_score'],
            'rel_level': data[image_name]['rel_level'],
            'target_area': target_area,
            'tot_area': tot_area,
            'context_area': context_area,
            'x_min': x_min,
            'y_min': y_min,
            'w': w,
            'h': h,
            'x_max': x_max,
            'y_max': y_max,
            'attn_over_target': attn_over_target_list,
            'attn_over_target_max': attn_over_target_max_list,
            'attn_over_context': attn_over_context_list,
            'attn_over_context_max': attn_over_context_max_list,
            'tot_attn': tot_attn_list,
            'attn_over_image_max': attn_over_image_max_list
        })

        # Final cleanup
        del llm_attn_matrix
        del att_on_whole_image
        gc.collect()
        torch.cuda.empty_cache()

# Convert the list to a DataFrame
results_df = pd.DataFrame(results_list)

# Define the file path
output_file = os.path.join(output_dir, "results.csv")

# Save the DataFrame to a CSV file
results_df.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")
