import os
import json
from config import *

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
    