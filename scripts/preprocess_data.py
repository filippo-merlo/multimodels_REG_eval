# preprocess_data.py
from config import *
import json
import os
from pprint import pprint

with open(dataset_path, 'r') as f:
    dataset = json.load(f)


extensions = ['.jpg']

all_image_paths = []

# Loop through each folder in the root image directory
for folder_name in os.listdir(image_dir_path):
    folder_path = os.path.join(image_dir_path, folder_name)

    # Check if the folder path is actually a directory
    if os.path.isdir(folder_path):
        # Loop through each file in the folder
        for image_name in os.listdir(folder_path):
            # Check if file extension matches
            if any(image_name.lower().endswith(ext) for ext in extensions):
                # Append full path to the final list
                all_image_paths.append(os.path.join(folder_path, image_name))

pprint(len(all_image_paths))
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
    