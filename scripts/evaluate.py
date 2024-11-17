# evaluate.py
import torch
import json
from models_scripts import load_model  # Assuming a load_model function is defined to load your model
from data import load_dataset, get_images_names_path
from utils import log_metrics, add_grey_background_and_rescale_bbox, add_gaussian_noise_in_bbox
from config import *

def evaluate(model_name, data, images_n_p, device):
    # Load the model
    model, generate = load_model(model_name, device, model_dir, cache_dir)
    
    # Run evaluation
    noise_levels = [0.0, 0.5, 1.0]
    results = {
        '0.0_target': [],
        '0.0_output': [],
        '0.5_target': [],
        '0.5_output': [],
        '1.0_target': [],
        '1.0_output': []
    }

    for noise_level in noise_levels:
        for image_name, image_path in images_n_p.items():
            # Exclude images that has been filtered out by the LLAVA filter
            if data[image_name]['excluded']:
                continue
            bbox = data[image_name]['target_bbox']
            
            if 'original' in image_name:
                target = data[image_name]['target']
            if 'clean' in image_name:
                target = 'nothing'
            else:
                target = data[image_name]['swapped_object']
            
            # get the image with a grey background and the bounding box rescaled
            image, bbox = add_grey_background_and_rescale_bbox(image_path, bbox)

            # get the image with the corresponding noise level in the roi
            image = add_gaussian_noise_in_bbox(image, bbox, noise_level)

            # get the input for the model that is
            # prompt with the right notation for indicating the target area
            # the image
            # eventually the bounding box if the model accepts it
            
            output = generate(model, image, bbox)
            print('****************')
            print('target:', target)
            print('output:', output)
            print('\n')

            results[str(noise_level)+'_target'].append(target)
            results[str(noise_level)+'_output'].append(output)


    # Calculate metrics
    #metrics = calculate_metrics(results, data)
    metrics = None
    
    # Log results
    #log_metrics(model_name, metrics)
    return metrics

def calculate_metrics(results, data):
    # Placeholder for metric calculation logic
    return 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="The name of the model to evaluate")
    parser.add_argument("--device", type=str, help="The device to run evaluation on, e.g., 'cuda:0'")
    args = parser.parse_args()

    # python evaluate.py --model_name Salesforce/xgen-mm-phi3-mini-instruct-singleimg-r-v1.5 --device cuda
    # python evaluate.py --model_name Salesforce/xgen-mm-phi3-mini-instruct-r-v1 --device cuda
    # python evaluate.py --model_name 'microsoft/kosmos-2-patch14-224' --device cuda
    # python evaluate.py --model_name 'allenai/Molmo-7B-O-0924' --device cuda

    # Load data
    data = load_dataset()
    images_n_p = get_images_names_path()

    # Evaluate
    metrics = evaluate(args.model_name, data, images_n_p, args.device)
    '''
    # Save results
    with open(f'outputs/results/{args.model_name}_metrics.json', 'w') as f:
        json.dump(metrics, f)
    '''
