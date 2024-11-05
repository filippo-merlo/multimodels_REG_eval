# utils.py
import json
import os

def log_metrics(model_name, metrics):
    log_path = f"outputs/logs/{model_name}_log.json"
    with open(log_path, 'w') as f:
        json.dump(metrics, f)
    print(f"Logged metrics for {model_name}")


from PIL import Image
import numpy as np

def add_gaussian_noise_in_bbox(image_path, bbox, noise_level=0.0):
    # Add noise to the image within the bounding box
    image = Image.open(image_path)
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


def get_input(model_name, image, bbox):
    # Placeholder for input preparation logic
    return