# run_all_evaluations.py
import subprocess
from scripts.manage_gpu import free_gpu_memory
from scripts.data import load_dataset, get_images_names_path


# model list 
model_list = [
    'cyan2k/molmo-7B-O-bnb-4bit',
    'Salesforce/xgen-mm-phi3-mini-instruct-singleimg-r-v1.5',
    'Salesforce/xgen-mm-phi3-mini-instruct-r-v1',
    'microsoft/kosmos-2-patch14-224',
    'Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8',
    'llava-hf/llava-onevision-qwen2-0.5b-si-hf',
]


device = "cuda"  # Specify GPU device

for model_name in model_list:
    print(f"Evaluating {model_name}...")
    # Load data
    subprocess.run(["python", "scripts/evaluate.py", 
                    "--model_name", model_name, 
                    "--device", device],
                    check=True)

    # Free GPU memory
    free_gpu_memory()
    print(f"Completed evaluation for {model_name}")