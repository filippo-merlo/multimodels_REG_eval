# evaluate.py
from multimodels_REG_eval.scripts.models_scripts import load_model  # Assuming a load_model function is defined to load your model
from data import load_dataset, get_images_names_path
from utils import log_metrics, add_grey_background_and_rescale_bbox, add_gaussian_noise_in_bbox, add_gaussian_noise_outside_bbox, get_image_patch
from config import *
from metrics import compute_metrics

import pandas as pd
import os
from PIL import Image

## Add the directory to sys.path
#directory = os.path.abspath("/home/filippo.merlo/caption_evaluation")
#sys.path.insert(0, directory)

def evaluate(model_name, data, images_n_p, device):
    # Load the model
    model, generate = load_model(model_name, device, model_dir, cache_dir)

    # Initialize a list to collect evaluation results
    evaluation_results = []
    
    # Run evaluation
    noise_levels = [0.0, 0.5, 1.0]
    conditions = ['target_noise','context_noise','all_noise']

    for condition in conditions:
        for noise_level in noise_levels:
            for image_name, image_path in list(images_n_p.items()):
                try:    
                    if noise_level == 0.0 and condition != 'target_noise':
                        continue
                
                    # Exclude images that has been filtered out by the LLAVA filter
                    if data[image_name]['excluded']:
                        continue
                    
                    bbox = data[image_name]['target_bbox']
                    
                    if '_original.jpg' in image_name:
                        target = data[image_name]['target'].replace('_', ' ')
                    elif '_clean.jpg' in image_name:
                        continue # skip clean images
                    else:
                        target = data[image_name]['swapped_object'].replace('_', ' ')

                    # get the image with a grey background and the bounding box rescaled
                    image, bbox = add_grey_background_and_rescale_bbox(image_path, bbox)
                    image_patch = get_image_patch(image, bbox)
                    temporary_save_path_image_patch = os.path.join(temporary_save_dir,f'image_patch_{image_name}')
                    if not os.path.exists(temporary_save_path_image_patch):
                        image_patch.save(temporary_save_path_image_patch)
                    # load patch
                    image_patch = Image.open(temporary_save_path_image_patch)


                    # get the image with the corresponding noise level in the roi
                    if condition == 'target_noise':
                        image = add_gaussian_noise_in_bbox(image, bbox, noise_level)
                    elif condition == 'context_noise':
                        image = add_gaussian_noise_outside_bbox(image, bbox, noise_level)
                    elif condition == 'all_noise':
                        image = add_gaussian_noise_in_bbox(image, bbox, noise_level)
                        image = add_gaussian_noise_outside_bbox(image, bbox, noise_level)

                    # get the input for the model that is
                    # prompt with the right notation for indicating the target area
                    # the image
                    # eventually the bounding box if the model accepts it
                    
                    raw_output = generate(model, image, bbox)

                    # format output
                    output = raw_output.lstrip().lower()
                    # Remove "a " or "an " if the string starts with either
                    if output.lower().startswith("a "):
                        output = output[2:]
                    elif output.lower().startswith("an "):
                        output = output[3:]

                    common_prefix = 'A photo depicts '

                    if target[0] in ['a', 'e', 'i', 'o', 'u']:
                        complete_prefix = common_prefix + 'an '
                        long_target = complete_prefix + target
                    else:
                        complete_prefix = common_prefix + 'a '
                        long_target = complete_prefix + target

                    if output[0] in ['a', 'e', 'i', 'o', 'u']:
                        complete_prefix = common_prefix + 'an '
                        long_output = complete_prefix + output
                    else:
                        complete_prefix = common_prefix + 'a '
                        long_output = complete_prefix + output
                    
                    
                    ref_clip_score, text_similarity_score = compute_metrics(output,target,image_patch)
                    long_caption_ref_clip_score, long_caption_text_similarity_score = compute_metrics(long_output,long_target,image_patch)

                    #scores = compute_ensembeval_score(candidates, references, image_paths)
                    # Where candidates is a list of captions, references is a list of lists of reference captions, image_paths is a list of strings with locations of images.
                    #scores = compute_ensembeval_score([str(output)],[[str(target)]],[temporary_save_path_image_patch], weights=weights)
                    print('****************')
                    print(image_name)
                    print('target:', target)
                    print('long_target:', long_target)
                    print('output:', output)
                    print('long_output:', long_output)
                    print('\n')
                    print('ref_clip_score:',ref_clip_score)
                    print('text_similarity_score:',text_similarity_score)
                    print('long_caption_ref_clip_score:',long_caption_ref_clip_score)
                    print('long_caption_text_similarity_score:',long_caption_text_similarity_score)
                    print('\n')

                    # Append the results
                    evaluation_results.append({
                        'model_name': model_name,
                        'image_name': image_name,
                        'noise_level': noise_level,
                        'condition': condition,
                        'target': target,
                        'long_target': long_target,
                        'raw_output': raw_output,
                        'output': output,
                        'long_output': long_output,
                        'scores': ref_clip_score,
                        'text_similarity_scores': text_similarity_score,
                        'long_caption_scores': long_caption_ref_clip_score,
                        'long_caption_text_similarity_scores': long_caption_text_similarity_score,
                        'scene': data[image_name]['scene'],
                        'rel_score': data[image_name]['rel_score'],
                        'rel_level': data[image_name]['rel_level']
                    })
                except Exception as e:
                    print(f"Error processing image {image_name}: {e}")
                    continue
    
    results_df = pd.DataFrame(evaluation_results)
    return results_df



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="The name of the model to evaluate")
    parser.add_argument("--device", type=str, help="The device to run evaluation on, e.g., 'cuda:0'")
    args = parser.parse_args()
    
    # python evaluate.py --model_name 'allenai/Molmo-7B-D-0924' --device cuda
    # python evaluate.py --model_name 'Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5' --device cuda
    # python evaluate.py --model_name 'microsoft/kosmos-2-patch14-224' --device cuda
    # python evaluate.py --model_name 'Qwen/Qwen2-VL-7B-Instruct' --device cuda
    # python evaluate.py --model_name 'llava-hf/llava-onevision-qwen2-0.5b-si-hf' --device cuda
    # python evaluate.py --model_name 'llava-hf/llava-onevision-qwen2-7b-ov-hf' --device cuda

    # Load data
    data = load_dataset(dataset_path)
    images_n_p = get_images_names_path(images_path)

    # Evaluate
    results_df = evaluate(args.model_name, data, images_n_p, args.device)
    results_df.to_csv(f"{output_dir}/{args.model_name.replace('/', '_')}_results.csv")

