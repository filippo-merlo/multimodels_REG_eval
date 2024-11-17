# utils.py
import json
import os

def log_metrics(model_name, metrics):
    log_path = f"outputs/logs/{model_name}_log.json"
    with open(log_path, 'w') as f:
        json.dump(metrics, f)
    print(f"Logged metrics for {model_name}")


from PIL import Image, ImageOps
import numpy as np

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
