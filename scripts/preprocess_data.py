import os
import json
from tqdm import tqdm
from PIL import Image
from config import dataset_path, images_path, new_images_path

# Load dataset
with open(dataset_path, 'r') as f:
    dataset = json.load(f)

# Supported image file extensions
extensions = ['.jpg']

# Prepare dictionaries to store image paths and IDs
all_image_paths = {}
only_ids = set()

# Collect image paths and IDs
for folder_name in os.listdir(images_path):
    folder_path = os.path.join(images_path, folder_name)
    if os.path.isdir(folder_path):
        for image_name in os.listdir(folder_path):
            if any(image_name.lower().endswith(ext) for ext in extensions):
                image_id = image_name.split('_')[0]
                only_ids.add(image_id)
                all_image_paths[image_name] = os.path.join(folder_path, image_name)

only_ids = list(only_ids)

# Ensure the output directory exists
os.makedirs(new_images_path, exist_ok=True)

# Process each ID in `only_ids`
for id in tqdm(only_ids):

    # Initialize details for the original image
    original_image_path = None
    original_bbox = None
    original_size = None

    # Identify the original image and its properties
    for image_name, image_path in all_image_paths.items():
        if id in image_name and 'original' in image_name:
            original_image_path = image_path
            original_name = image_name
            original_image = Image.open(original_image_path)
            original_size = original_image.size
            original_image.save(os.path.join(new_images_path, original_name))  # Save original
            original_bbox = dataset[original_name]['target_bbox']
            break

    if not original_image_path or not original_size:
        continue  # Skip if no original image is found for the ID

    original_width, original_height = original_size

    # Process non-original images
    for image_name, image_path in all_image_paths.items():
        if id in image_name and 'original' not in image_name:
            image = Image.open(image_path)
            width, height = image.size
            width_ratio, height_ratio = original_width / width, original_height / height

            resized_image = image.resize((original_width, original_height), Image.LANCZOS)
            
            if 'clean' in image_name:
                dataset[image_name] = dataset[original_name]
            else:
                x, w, y, h = dataset[image_name]['target_bbox']
                new_box = [
                    int(x * width_ratio), int(w * width_ratio),
                    int(y * height_ratio), int(h * height_ratio)
                ]
                dataset[image_name]['target_bbox'] = new_box

            # Save resized image
            resized_image.save(os.path.join(new_images_path, image_name))

# Save the updated dataset
with open(os.path.join(dataset_path, 'final_dataset_resized.json'), 'w') as f:
    json.dump(dataset, f)

#%%
def load_data():
    # Load and preprocess data
    data_path = "data/processed/benchmark_data.json"
    if not os.path.exists(data_path):
        preprocess_data()
    with open(data_path, 'r') as f:
        return json.load(f)

def preprocess_data():
    # Placeholder for actual data preprocessing
    raw_data_path = "data/raw/raw_data.json"
    processed_data_path = "data/processed/benchmark_data.json"
    with open(raw_data_path, 'r') as f:
        raw_data = json.load(f)
    
    processed_data = raw_data  # Apply processing here
    with open(processed_data_path, 'w') as f:
        json.dump(processed_data, f)
    