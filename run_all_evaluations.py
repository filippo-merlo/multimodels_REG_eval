# run_all_evaluations.py
import subprocess
from scripts.manage_gpu import free_gpu_memory
from scripts.data import load_dataset, get_images_names_path


# model list 
model_list = [
    'Salesforce/xgen-mm-phi3-mini-instruct-singleimg-r-v1.5',
    'Salesforce/xgen-mm-phi3-mini-instruct-r-v1',
    'microsoft/kosmos-2-patch14-224',
    'cyan2k/molmo-7B-D-bnb-4bit',
    'Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8',
    'llava-hf/llava-onevision-qwen2-0.5b-si-hf',
]


device = "cuda:0"  # Specify GPU device

for model_name in model_list:
    print(f"Evaluating {model_name}...")
    # Load data

    # Run evaluation as a subprocess to manage memory usage
    for model_name in model_list:
        subprocess.run(["python", "scripts/evaluate.py", 
                        "--model_name", model_name, 
                        "--device", device],
                       check=True)

        # Free GPU memory
        free_gpu_memory()
        
        print(f"Completed evaluation for {model_name}")