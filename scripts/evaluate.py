# evaluate.py
import torch
import json
from models import load_model  # Assuming a load_model function is defined to load your model
from scripts.adjust_dataset_2_delete import load_data
from scripts.utils import log_metrics

def evaluate(model_name, data, device):
    # Load the model
    model = load_model(model_name, device)
    
    # Run evaluation
    results = []
    for batch in data:
        output = model(batch)
        results.append(output)

    # Calculate metrics
    metrics = calculate_metrics(results, data)
    
    # Log results
    log_metrics(model_name, metrics)
    return metrics

def calculate_metrics(results, data):
    # Placeholder for metric calculation logic
    return {"accuracy": sum([1 for r, d in zip(results, data) if r == d]) / len(data)}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="The name of the model to evaluate")
    parser.add_argument("device", type=str, help="The device to run evaluation on, e.g., 'cuda:0'")
    args = parser.parse_args()
    
    # Load data
    data = load_data()
    
    # Evaluate
    metrics = evaluate(args.model_name, data, args.device)
    
    # Save results
    with open(f'outputs/results/{args.model_name}_metrics.json', 'w') as f:
        json.dump(metrics, f)
