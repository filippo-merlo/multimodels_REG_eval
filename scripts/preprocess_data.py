import os
import json
from config import data_dir_path, dataset_path, images_path

def load_dataset():
    # Load and preprocess data
    with open(dataset_path, 'r') as f:
        return json.load(f)

def get_images_names_path():
    # Load and preprocess images
    images_n_p = {}
    for filename in os.listdir(images_path):
        if filename.endswith('.jpg'):
            images_n_p[filename] = os.path.join(images_path, filename)
    return images_n_p


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

dataset = load_dataset()
images = get_images_names_path()

i = 0
os.makedirs(os.path.dirname(os.path.join(data_dir_path,'noisy_images')))
for image_name, image_path in images.items():
    i += 1
    if i > 100:
        break
    bbox = dataset[image_name]['target_bbox']
    noisy_image = add_gaussian_noise_in_bbox(image_path, bbox, noise_level=0.5)
    noisy_image.save(os.path.join(data_dir_path, f'noisy_images/{image_name}'))