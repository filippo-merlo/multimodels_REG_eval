# utils.py
import json
import os

def log_metrics(model_name, metrics):
    log_path = f"outputs/logs/{model_name}_log.json"
    with open(log_path, 'w') as f:
        json.dump(metrics, f)
    print(f"Logged metrics for {model_name}")