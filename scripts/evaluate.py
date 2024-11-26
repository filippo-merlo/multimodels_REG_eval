# evaluate.py
import torch
import json
from models_scripts import load_model  # Assuming a load_model function is defined to load your model
from data import load_dataset, get_images_names_path
from utils import log_metrics, add_grey_background_and_rescale_bbox, add_gaussian_noise_in_bbox, get_image_patch
from config import *
from transformers import CLIPModel, CLIPProcessor
from torch.nn.functional import cosine_similarity
from metrics.ensembeval_score import compute_ensembeval_score
import os

os.environ['HF_HOME'] = cache_dir

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
        '1.0_output': [],
    }

    for noise_level in noise_levels:
        for image_name, image_path in list(images_n_p.items())[:2]:
            print(image_name)
            # Exclude images that has been filtered out by the LLAVA filter
            if data[image_name]['excluded']:
                continue
            
            bbox = data[image_name]['target_bbox']
            
            if '_original.jpg' in image_name:
                target = data[image_name]['target'].replace('_', ' ')
            elif '_clean.jpg' in image_name:
                target = 'nothing'
            else:
                target = data[image_name]['swapped_object'].replace('_', ' ')

            # get the image with a grey background and the bounding box rescaled
            image, bbox = add_grey_background_and_rescale_bbox(image_path, bbox)

            # get the image with the corresponding noise level in the roi
            image = add_gaussian_noise_in_bbox(image, bbox, noise_level)

            image_patch = get_image_patch(image, bbox)
            temporary_save_path_image_patch = os.path.join(temporary_save_dir,'image_patch.jpg')
            image_patch.save(temporary_save_path_image_patch)


            # get the input for the model that is
            # prompt with the right notation for indicating the target area
            # the image
            # eventually the bounding box if the model accepts it
            
            output = generate(model, image, bbox).lower()
            print('****************')
            print('target:', target)
            print('output:', output)
            print('\n')

            #print(ref_clip_score(str(target), str(formatted_output), image_patch))
            #scores = compute_ensembeval_scorecandidates, references, image_paths, weights=your_weights)
            scores = compute_ensembeval_score([str(output)],[str(target)],[temporary_save_path_image_patch])
            print(scores)

            results[str(noise_level)+'_target'].append(target)
            results[str(noise_level)+'_output'].append(output.replace('_', ' '))


    # Calculate metrics
    #metrics = calculate_metrics(results, data)
    metrics = None
    
    # Log results
    #log_metrics(model_name, metrics)
    return metrics


# Load CLIP model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(model_name)

def compute_image_embedding(image):
    inputs = clip_processor(
        text=[""],
        images=image,
        return_tensors="pt",
        padding=True
    ).to(clip_model.device)
   
    with torch.no_grad():
        image_embeds = clip_model.get_image_features(inputs["pixel_values"])
    image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)
    return image_embeds
   
def compute_text_embedding(text):
    """Compute text embeddings for a text using CLIP."""
    inputs = clip_processor(
        text=[text],
        images=None,
        return_tensors="pt",
        padding=True
    ).to(clip_model.device)
    with torch.no_grad():
        text_embeds = clip_model.get_text_features(inputs["input_ids"], inputs["attention_mask"])
    text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)
    return text_embeds

def ref_clip_score(target, generated_text, image):
    """
    Compute RefCLIPScore for a candidate text given references and a visual embedding.
    
    Args:
    - candidate (str): The candidate caption.
    - references (list of str): The list of reference captions.
    - visual_embedding (torch.Tensor): The CLIP image embedding (1D vector).
    
    Returns:
    - score (float): The RefCLIPScore value.
    """

    # Compute candidate and reference embeddings
    target_embedding = compute_text_embedding(target).squeeze(0)
    reference_embeddings = compute_text_embedding(generated_text)
    image_embedding = compute_image_embedding(image)

    # Compute CLIP-S (cosine similarity between candidate and image embedding)
    clip_s = (target_embedding @ image_embedding.T).squeeze()

    # Compute max reference similarity (cosine similarity between candidate and references)
    ref_sims = (target_embedding @ reference_embeddings.T).squeeze()
    print(ref_sims)
    max_ref_sim = torch.max(ref_sims).item()

    # Compute harmonic mean of CLIP-S and max reference similarity
    if clip_s + max_ref_sim > 0:
        ref_clip_s = 2 * clip_s * max_ref_sim / (clip_s + max_ref_sim)
    else:
        ref_clip_s = 0.0

    return ref_clip_s


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="The name of the model to evaluate")
    parser.add_argument("--device", type=str, help="The device to run evaluation on, e.g., 'cuda:0'")
    args = parser.parse_args()

    # python evaluate.py --model_name Salesforce/xgen-mm-phi3-mini-instruct-singleimg-r-v1.5 --device cuda
    # python evaluate.py --model_name Salesforce/xgen-mm-phi3-mini-instruct-r-v1 --device cuda
    # python evaluate.py --model_name 'microsoft/kosmos-2-patch14-224' --device cuda
    # python evaluate.py --model_name 'cyan2k/molmo-7B-O-bnb-4bit' --device cuda
    # python evaluate.py --model_name 'Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8' --device cuda 
    # python evaluate.py --model_name 'THUDM/cogvlm2-llama3-chat-19B-int4' --device cuda #4.45.0
    # python evaluate.py --model_name "llava-hf/llava-onevision-qwen2-0.5b-si-hf" --device cuda 


    # Load data
    data = load_dataset(dataset_path)
    images_n_p = get_images_names_path(images_path)

    # Evaluate
    metrics = evaluate(args.model_name, data, images_n_p, args.device)
    '''
    # Save results
    with open(f'outputs/results/{args.model_name}_metrics.json', 'w') as f:
        json.dump(metrics, f)
    '''
