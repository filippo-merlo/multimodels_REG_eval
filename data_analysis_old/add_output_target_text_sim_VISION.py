#%%
import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from pprint import pprint

tqdm.pandas()


# --- Load the data ---
dataset_path = '/home/fmerlo/data/sceneregstorage/attn_eval_output/results_att_deployment_VISIONS_complete.csv'

df = pd.read_csv(dataset_path)
df


#%% Add output - target similarity scores
from transformers import CLIPModel, CLIPProcessor
import torch

device = "cuda"

model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(model_name)

def compute_text_embedding(text):
    inputs = clip_processor(
        text=[text] if isinstance(text, str) else text,
        return_tensors="pt",
        padding=True
    ).to(clip_model.device)
    
    with torch.no_grad():
        text_embeds = clip_model.get_text_features(inputs["input_ids"], inputs["attention_mask"])
    return text_embeds / text_embeds.norm(dim=1, keepdim=True)

# Remove "a " or "an " if the string starts with either
def add_prefix(scene):
    common_prefix = 'A photo depicts '
    if scene[0] in ['a', 'e', 'i', 'o', 'u']:
        return common_prefix + 'an '+ scene.replace('_', ' ')
    else:
        return common_prefix + 'a '+ scene.replace('_', ' ')
        
# cosine similarity: torch.nn.functional.cosine_similarity(e_1, e_2, dim=1).item()

def compute_cosine_similarity(row):
    text_embed_1 = compute_text_embedding(add_prefix(row.output_clean))
    text_embed_2 = compute_text_embedding(add_prefix(row.image_name.split("_")[0]))
    print(f"Output text: {row.output_clean}"
          f"\nTarget text: {row.image_name.split('_')[0]}")
    return torch.nn.functional.cosine_similarity(text_embed_1, text_embed_2, dim=1).item()

#%%
df['output_scene_text_similarity_scores'] = [compute_cosine_similarity(row) for row in tqdm(df.itertuples(index=False), total=len(df))]
#%%
df
#%%
# Save the updated DataFrame to a new CSV file
output_path = '/home/fmerlo/data/sceneregstorage/attn_eval_output/results_att_deployment_VISIONS_complete_wscores.csv'
df.to_csv(output_path, index=False)
print(f"Saved updated dataset to: {output_path}")