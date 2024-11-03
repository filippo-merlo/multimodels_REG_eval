# preprocess_data.py
from config import *
import json
import os
from pprint import pprint

with open(dataset_path, 'r') as f:
    dataset = json.load(f)

image_names_list = []
image_folder_names_list = os.listdir(image_dir_path)
for folder_name in image_folder_names_list:
    folder_path = os.path.join(image_dir_path, folder_name)
    image_names_list = os.listdir(folder_path)
    for image_name in image_names_list:
        if image_name.endswith('.jpg'):
            image_names_list.append(image_name)
pprint(image_names_list)

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
    