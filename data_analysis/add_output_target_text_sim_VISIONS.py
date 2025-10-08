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
file_path = '/home/fmerlo/data/sceneregstorage/attn_eval_output/results_att_deployment_VISIONS.csv'  # Update with your actual file path

df = pd.read_csv(file_path)
# Preview first 5 rows
print("Head of the DataFrame:")
df.head()
#%%
# Shape of the DataFrame
print("Shape (rows, columns):", df.shape, "\n")

# Column names and data types
print("Column Information:")
df.columns

#%%
df['rel_level'] = df['rel_level'].str.replace('c', 'Congruent', regex=True)
df['rel_level'] = df['rel_level'].str.replace('i', 'Incongruent', regex=True)
df['condition'] = df['condition'].str.replace('_noise', '', regex=True)
df['output_clean'] = df['output_text'].str.replace(r'<\|im_end\|>', '', regex=True).str.replace(r'\.', '', regex=True).str.lower()
df.columns
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
    text_embed_2 = compute_text_embedding(add_prefix(row.target))
    return torch.nn.functional.cosine_similarity(text_embed_1, text_embed_2, dim=1).item()

#%%
df['long_caption_text_similarity_scores'] = [compute_cosine_similarity(row) for row in tqdm(df.itertuples(index=False), total=len(df))]

#%%
# Save the updated DataFrame to a new CSV file
output_path = '/home/fmerlo/data/sceneregstorage/attn_eval_output/results_att_deployment_VISIONS_wscores.csv'
df.to_csv(output_path, index=False)
print(f"Saved updated dataset to: {output_path}")