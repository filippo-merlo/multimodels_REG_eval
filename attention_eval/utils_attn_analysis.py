from io import BytesIO
import requests
import cv2
import numpy as np
from PIL import Image
import json
import torch
import pandas as pd
import os

# Utils for the task

def log_metrics(model_name, metrics):
    log_path = f"outputs/logs/{model_name}_log.json"
    with open(log_path, 'w') as f:
        json.dump(metrics, f)
    print(f"Logged metrics for {model_name}")

# Function to monitor GPU memory
def log_gpu_usage():
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1e6} MB")
    print(f"Cached memory: {torch.cuda.memory_reserved() / 1e6} MB")

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

def rescale_image_add_grey_background_and_rescale_bbox(image_path, bbox, max_width=640):
    """
    Resizes the input image to have a maximum width of 640 px (keeping aspect ratio),
    rescales the bounding box accordingly, adds a grey square background,
    and centers the resized image on it.

    Args:
        image_path (str): Path to input image.
        bbox (tuple): A tuple describing the bounding box in the format (x, y, w, h).

    Returns:
        PIL.Image.Image: The new image with a grey background.
        tuple: The rescaled bounding box in the format (x, y, w, h).
    """
    image = Image.open(image_path)

    # --- Step 1: Resize the image if width > 640 px ---
    original_width, original_height = image.width, image.height
    x, y, w, h = bbox

    if original_width > max_width:
        scale_factor = max_width / original_width
        new_width = max_width
        new_height = int(original_height * scale_factor)

        # Resize the image
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Rescale the bounding box
        x = int(x * scale_factor)
        y = int(y * scale_factor)
        w = int(w * scale_factor)
        h = int(h * scale_factor)

    # --- Step 2: Add grey square background ---
    max_dim = max(image.width, image.height)
    new_size = (max_dim, max_dim)

    grey_background = Image.new("RGB", new_size, color=(128, 128, 128))

    # Center the image on the background
    offset_x = (max_dim - image.width) // 2
    offset_y = (max_dim - image.height) // 2
    grey_background.paste(image, (offset_x, offset_y))

    # Adjust bounding box to new centered position
    new_bbox = (x + offset_x, y + offset_y, w, h)

    return grey_background, new_bbox, (new_width, new_height)

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


# Attention utils

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
            #torch.tensor([0.]),
        ))
        avged.append(vec / vec.sum())
    return torch.stack(avged).mean(dim=0)

def aggregate_vit_attention(attn, select_layer):
    layer = attn[select_layer]
    layer_attns = layer.squeeze(0)
    attns_per_head = layer_attns.mean(dim=0)
    result = attns_per_head / attns_per_head.sum(-1, keepdim=True)
    result = result.cpu()
    del layer, layer_attns, attns_per_head
    return result.cpu()

def aggregate_vit_attention_subtract_avg(attn, attn_avg, select_layer):
        # For single layer mode, return adjusted attention for the selected layer
        selected_layer = attn[select_layer]
        layer_attns = selected_layer.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        vec = attns_per_head[0:, 0:].cpu()
        vec_normalized = vec / vec.sum(-1, keepdim=True)
        #attn_avg = attn_avg[i] / attn_avg[i].sum(-1, keepdim=True) # good idea to normalize?
        return torch.relu(vec_normalized - attn_avg[select_layer])

def heterogenous_stack(vecs):
    '''Pad vectors with zeros then stack'''
    max_length = max(v.shape[0] for v in vecs)
    return torch.stack([
        torch.concat((v, torch.zeros(max_length - v.shape[0])))
        for v in vecs
    ])


# Image loading and processing

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

def get_images_names_path(images_path):
    # Load and preprocess images
    images_n_p = {}
    for filename in os.listdir(images_path):
        if filename.endswith('.jpg'):
            images_n_p[filename] = os.path.join(images_path, filename)
    return images_n_p

# Load Image And Load Data
def load_dataset(dataset_path):
    # Load and preprocess data
    with open(dataset_path, 'r') as f:
        return json.load(f)
    
def convert_to_bbox_format(df):
    """
    Convert corner coordinates in a DataFrame to [x1, y1, x2, y2] format.
    
    Args:
        df: Pandas DataFrame with columns ['scene_id', 'object', 'x', 'y']
            Each bounding box is defined by 4 corner points.

    Returns:
        A DataFrame with columns ['scene_id', 'object', 'bbox']
        where bbox is a list [x1, y1, x2, y2].
    """
    bbox_list = (
        df.groupby(['scene', 'obj'])
          .apply(lambda g: [g['x'].min(), g['y'].min(),
                            g['x'].max(), g['y'].max()])
          .reset_index(name='bbox')
    )
    return bbox_list

def get_bbox_data(bbox_dataset_path):
    """Load bounding box data from a CSV file without using any column as the index."""
    return convert_to_bbox_format(pd.read_csv(bbox_dataset_path))

def get_image_patch(image, bbox):
    """
    Extract a patch from an image based on a bounding box.
    
    Parameters:
        image (PIL.Image.Image): The input image.
        bbox (list or tuple): The bounding box in the format [x, y, w, h], 
                              where (x, y) is the top-left corner, and 
                              w and h are the width and height.
                              
    Returns:
        PIL.Image.Image: The cropped image patch.
    """
    x, y, w, h = bbox
    # Define the cropping box as (left, upper, right, lower)
    crop_box = (x, y, x + w, y + h)
    return image.crop(crop_box)
