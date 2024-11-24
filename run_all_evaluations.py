# run_all_evaluations.py
import subprocess
from scripts.manage_gpu import free_gpu_memory
from scripts.data import load_dataset, get_images_names_path
from scripts.config import *


device = "cuda:0"  # Specify GPU device

for model_name in model_list:
    print(f"Evaluating {model_name}...")
    # Load data
    data = load_dataset(dataset_path)
    images_n_p = get_images_names_path(images_path)

    # Run evaluation as a subprocess to manage memory usage
    for model_name in model_list:
        subprocess.run(["python", "scripts/evaluate.py", model_name, device])

        # Free GPU memory
        free_gpu_memory()
        
        print(f"Completed evaluation for {model_name}")