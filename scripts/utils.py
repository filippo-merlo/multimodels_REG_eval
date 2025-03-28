# utils.py
import json
from PIL import Image, ImageOps
import numpy as np


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

def normalize_box_N(bbox, image_width=1025, image_height=1025, N = 1000):
    return (
        min(int(bbox[0] * N / image_width), N),
        min(int(bbox[1] * N / image_height), N),
        min(int(bbox[2] * N / image_width), N),
        min(int(bbox[3] * N / image_height), N),
    )


def normalize_box_cogvlm(bbox, image_width=1025, image_height=1025, N = 1000):
    cohords =  [
        int(round(float(bbox[0] / image_width), 4) * N),
        int(round(float(bbox[1] / image_height), 4) * N),
        int(round(float(bbox[2] / image_width), 4) * N),
        int(round(float(bbox[3] / image_height), 4) * N),
    ]
    final = []
    for c in cohords: 
        s = str(c)
        if len(s) < 3:
            s = '0' + s
        final.append(s)
    return final

def convert_box(bbox):
    x, y, w, h = tuple(bbox) # Box coordinates are in (left, top, width, height) format
    return [x, y, x+w, y+h]

def convert_bbox_to_point(bbox):
    x, y, w, h = tuple(bbox)
    x1 = round(float(x + w/2),4)
    y1 = round(float(y + h/2),4)
    return (x1, y1)

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
