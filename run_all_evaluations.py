# run_all_evaluations.py
import subprocess
from scripts.manage_gpu import free_gpu_memory

models_to_evaluate = ["model1", "model2", "model3"]
device = "cuda:0"  # Specify GPU device

for model_name in models_to_evaluate:
    print(f"Evaluating {model_name}...")
    
    # Run evaluation as a subprocess to manage memory usage
    subprocess.run(["python", "scripts/evaluate.py", model_name, device])

    # Free GPU memory
    free_gpu_memory()
    
    print(f"Completed evaluation for {model_name}")