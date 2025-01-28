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




def aggregate_vit_attention(attn, select_layer):
    layer = attn[select_layer]
    layer_attns = layer.squeeze(0)
    attns_per_head = layer_attns.mean(dim=0)
    result = attns_per_head / attns_per_head.sum(-1, keepdim=True)
    result = result.cpu()
    del layer, layer_attns, attns_per_head
    return result.cpu()


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


data = load_dataset(dataset_path)
images_n_p = get_images_names_path(images_path)

noise_levels = [0.0]
conditions = ['target_noise']

vis_attn_matrix_average = []

for condition in conditions:
  for noise_level in noise_levels:
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

        del image_tensor, input_ids, outputs, model.get_vision_tower().image_attentions

        vis_attn_matrix_per_layers = []

        for layer in list(range(0,26)):

          vis_attn_matrix = aggregate_vit_attention(
              att_on_whole_image,
              select_layer=layer,
              all_prev_layers=False
          )
          vis_attn_matrix_per_layers.append(vis_attn_matrix)
          del vis_attn_matrix
        
        del att_on_whole_image

        if vis_attn_matrix_average == []:
            vis_attn_matrix_average = torch.stack(vis_attn_matrix_per_layers)
            for v in vis_attn_matrix_per_layers:
                del v
            del vis_attn_matrix_per_layers
        else:
            two_tensors = torch.stack([vis_attn_matrix_average, torch.stack(vis_attn_matrix_per_layers)])
            vis_attn_matrix_average = torch.mean(two_tensors, dim=0)
            for v in vis_attn_matrix_per_layers:
                del v
            del vis_attn_matrix_per_layers, two_tensors

        gc.collect()
        torch.cuda.empty_cache()

print(vis_attn_matrix_average)
print(vis_attn_matrix_average.size())
output_tensor_path = os.path.join(output_dir, "vis_attn_matrix_average.pt")
torch.save(vis_attn_matrix_average, output_tensor_path)
print(f"vis_attn_matrix_average saved at {output_tensor_path}")