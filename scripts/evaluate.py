# evaluate.py
import torch
import json
from models import load_model  # Assuming a load_model function is defined to load your model
from scripts.data import load_dataset, get_images_names_path
from scripts.utils import log_metrics, add_gaussian_noise_in_bbox

def evaluate(model_name, data, images_n_p, device):
    # Load the model
    model = load_model(model_name, device)
    
    # Run evaluation
    noise_levels = [0, 0.5, 1]
    results = {
        '0': [],
        '0.5': [],
        '1': []
    }
    for noise_level in noise_levels:
        for image_name, image_path in images_n_p.items():
            
            bbox = data[image_name]['target_bbox']
            input = get_input(model, image, target, bbox)
            output = model(input)
            results[str(noise_level)] = results[str(noise_level)].append(output)

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
    data = load_dataset()
    images_n_p = get_images_names_path()

    # Evaluate
    metrics = evaluate(args.model_name, data, images_n_p, args.device)
    
    # Save results
    with open(f'outputs/results/{args.model_name}_metrics.json', 'w') as f:
        json.dump(metrics, f)
