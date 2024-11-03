# preprocess_data.py
from config import *
import json
import os
from PIL import Image
from pprint import pprint

with open(dataset_path, 'r') as f:
    dataset = json.load(f)

print(dataset[list(dataset.keys())[0]])

extensions = ['.jpg']

all_image_paths = {}
only_ids = []
# Loop through each folder in the root image directory
for folder_name in os.listdir(images_path):
    folder_path = os.path.join(images_path, folder_name)

    # Check if the folder path is actually a directory
    if os.path.isdir(folder_path):
        # Loop through each file in the folder
        for image_name in os.listdir(folder_path):
            # Check if file extension matches
            if any(image_name.lower().endswith(ext) for ext in extensions):
                # Append full path to the final list
                only_ids.append(image_name.split('_')[0])
                all_image_paths[image_name] = os.path.join(folder_path, image_name)

only_ids = list(set(only_ids))

# Iterate through all ids
for id in only_ids[0]:

    # Get original image details
    for k in all_image_paths.keys():
        if id in k and 'original' in k:
            original_name = k
            original_path = all_image_paths[k]
            original_image = Image.open(original_path)
            original_width, original_height = original_image.size
            original_image.save(os.path.join(new_images_path, original_name))

    # Process non-original images
    for k in all_image_paths.keys():
        if id in k and 'original' not in k:
            not_original_name = k
            not_original_path = all_image_paths[k]
            
            # Open the non-original image
            image = Image.open(not_original_path)
            width, height = image.size
            
            # Calculate resizing ratios
            width_ratio = original_width / width
            height_ratio = original_height / height

            # Resize the image to match original dimensions
            resized_image = image.resize((original_width, original_height), Image.LANCZOS)

            # Get bbox
            x,w,y,h = dataset[id]['target_bbox'] 
            new_box = [int(x*width_ratio), int(w*width_ratio), int(y*height_ratio), int(h*height_ratio)]
            dataset[id]['target_bbox'] = new_box

            # Save or process the resized image as needed
            resized_image.save(os.path.join(new_images_path,not_original_name))

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
    