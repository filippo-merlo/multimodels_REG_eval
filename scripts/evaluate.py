# evaluate.py
import torch
import json
from models_scripts import load_model  # Assuming a load_model function is defined to load your model
from data import load_dataset, get_images_names_path
from utils import log_metrics, add_grey_background_and_rescale_bbox, add_gaussian_noise_in_bbox, get_image_patch
from config import *
from transformers import CLIPModel, CLIPProcessor
from torch.nn.functional import cosine_similarity
import pandas as pd
import sys
import os
from PIL import Image

## Add the directory to sys.path
#directory = os.path.abspath("/home/filippo.merlo/caption_evaluation")
#sys.path.insert(0, directory)
#
#from metrics.ensembeval_score import compute_ensembeval_score
#with open('/home/filippo.merlo/caption_evaluation/ensemble_weights.json', 'r') as fp:
#    weights = json.load(fp)


def evaluate(model_name, data, images_n_p, device):
    # Load the model
    model, generate = load_model(model_name, device, model_dir, cache_dir)

    # Initialize a list to collect evaluation results
    evaluation_results = []
    
    # Run evaluation
    noise_levels = [0.0, 0.5, 1.0]

    for noise_level in noise_levels:
        for image_name, image_path in list(images_n_p.items()):

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

            # get the image with the corresponding noise level in the roi
            image = add_gaussian_noise_in_bbox(image, bbox, noise_level)

            image_patch = get_image_patch(image, bbox)
            temporary_save_path_image_patch = os.path.join(temporary_save_dir,f'image_patch_{image_name}')
            if noise_level == 0.0:
                image_patch.save(temporary_save_path_image_patch)
            # load pathc
            image_patch = Image.open(temporary_save_path_image_patch)

            # get the input for the model that is
            # prompt with the right notation for indicating the target area
            # the image
            # eventually the bounding box if the model accepts it
            
            output = generate(model, image, bbox).lower()

            # format output
            output = output.lstrip()
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
                'target': target,
                'long_target': long_target,
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
    results_df = pd.DataFrame(evaluation_results)

    return results_df


# Load CLIP model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(model_name)

def compute_image_embedding(image):
    inputs = clip_processor(
        text=[""],  # Placeholder for image-only processing
        images=image,
        return_tensors="pt",
        padding=True
    ).to(clip_model.device)
    
    with torch.no_grad():
        image_embeds = clip_model.get_image_features(inputs["pixel_values"])
    return image_embeds / image_embeds.norm(dim=1, keepdim=True)


def compute_text_embedding(text):
    inputs = clip_processor(
        text=[text] if isinstance(text, str) else text,
        return_tensors="pt",
        padding=True
    ).to(clip_model.device)
    
    with torch.no_grad():
        text_embeds = clip_model.get_text_features(inputs["input_ids"], inputs["attention_mask"])
    return text_embeds / text_embeds.norm(dim=1, keepdim=True)


def clip_score(c_embedding: torch.Tensor, v_embedding: torch.Tensor, w: float = 2.5) -> float:
    """
    Compute the CLIP-Score for a given candidate caption and image using PyTorch.
    
    Args:
        c_embedding (torch.Tensor): CLIP embedding of the candidate caption (1D tensor).
        v_embedding (torch.Tensor): CLIP embedding of the image (1D tensor).
        w (float): Weight scaling factor (default is 2.5).
        
    Returns:
        float: The CLIP-Score.
    """
    # Compute cosine similarity
    cos_similarity = torch.nn.functional.cosine_similarity(c_embedding, v_embedding, dim=1)
    # Apply rescaling and max(0, cos_similarity)
    return w * max(cos_similarity.item(), 0)

def refclip_score(
    c_embedding: torch.Tensor,
    r_embeddings: list[torch.Tensor],
    v_embedding: torch.Tensor,
    w: float = 2.5
) -> float:
    """
    Compute the RefCLIPScore for a given candidate caption, image, and reference captions using PyTorch.
    
    Args:
        c_embedding (torch.Tensor): CLIP embedding of the candidate caption (1D tensor).
        r_embeddings (List[torch.Tensor]): List of CLIP embeddings of reference captions (1D tensors).
        v_embedding (torch.Tensor): CLIP embedding of the image (1D tensor).
        w (float): Weight scaling factor (default is 2.5).
        
    Returns:
        float: The RefCLIPScore.
    """
    # Compute CLIP-S(c, v)
    clip_s = clip_score(c_embedding, v_embedding, w=w)
    
    # Compute max cosine similarity between candidate and references
    max_ref_similarity = max(
        torch.nn.functional.cosine_similarity(c_embedding, r, dim=1).item()
        for r in r_embeddings
    )
    
    # Compute harmonic mean
    if clip_s > 0 and max_ref_similarity > 0:
        return 2 * clip_s * max_ref_similarity / (clip_s + max_ref_similarity)
    else:
        return 0.0
    
def compute_metrics(output, target, image_patch):
    # Compute embeddings
    output_embedding = compute_text_embedding(output)
    target_embedding = compute_text_embedding(target)
    image_embedding = compute_image_embedding(image_patch)
    
    # Compute scores
    ref_clip_score = refclip_score(output_embedding, [target_embedding], image_embedding)
    text_similarity_score = torch.nn.functional.cosine_similarity(output_embedding, target_embedding, dim=1).item()
    
    return ref_clip_score, text_similarity_score
    


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
    results_df = evaluate(args.model_name, data, images_n_p, args.device)
    results_df.to_csv(f"{output_dir}/{args.model_name.replace('/', '_')}_results.csv")

