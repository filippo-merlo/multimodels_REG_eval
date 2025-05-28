#%%
import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from pprint import pprint

tqdm.pandas()

separator = "\n\n##################################################\n##################################################\n\n"

# --- Load the data ---
file_path = '/home/fmerlo/data/sceneregstorage/eval_output/dataset_final_complete.csv'  # Update with your actual file path
df = pd.read_csv(file_path)

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
    #print('###')
    #print(add_prefix(row.output_clean))
    #print(add_prefix(row.original_target))
    text_embed_1 = compute_text_embedding(add_prefix(row.original_target))
    text_embed_2 = compute_text_embedding(add_prefix(row.output_clean))
    sim = torch.nn.functional.cosine_similarity(text_embed_1, text_embed_2, dim=1).item()
    #print('Similarity:', sim)
    return sim

#%%
id_original_target = {}
for name in df['image_name'].unique():
    if 'original.jpg' in name:
        id = name.split('_')[0]
        id_original_target[id] = str(df.loc[df['image_name'] == name, 'target'].values[0])

original_target_list = []
for name in df['image_name']:
     id = name.split('_')[0]
     original_target_list.append(id_original_target[id])
#%%
df['original_target'] = original_target_list
#%%
len(df)
#%%
df['original_target_output_similarity'] = [compute_cosine_similarity(row) for row in tqdm(df.itertuples(index=False), total=len(df))]

#%%
# Save the updated DataFrame to a new CSV file
output_path = '/home/fmerlo/data/sceneregstorage/eval_output/dataset_final_final_complete.csv'
df.to_csv(output_path, index=False)
print(f"Saved updated dataset to: {output_path}")