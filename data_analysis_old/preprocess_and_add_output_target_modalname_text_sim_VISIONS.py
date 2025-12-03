#%%
import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pprint import pprint

tqdm.pandas()

SEPARATOR = "\n\n" + "#" * 50 + "\n" + "#" * 50 + "\n\n"

# Load data
DATASET_PATH = '/home/fmerlo/data/sceneregstorage/attn_eval_output/results_att_deployment_VISIONS.csv'
df = pd.read_csv(DATASET_PATH)

print("Head of the DataFrame:")
print(df.head())

print(SEPARATOR)
print(f"Shape (rows, columns): {df.shape}\n")
print("Column names:")
pprint(df.columns.tolist())


# Inspect unique values
for col in df.columns:
    print(SEPARATOR)
    print(f"Column: {col}")
    print(df[col].unique())


# Preprocessing
df['Condition'] = df['condition'].str.replace('_noise', '', regex=True)
df = df.drop(columns=['condition'])

df['output_clean'] = (
    df['output_text']
    .str.replace(r'<\|im_end\|>', '', regex=True)
    .str.replace(r'\.', '', regex=True)
    .str.lower()
)

df['Rel. Level'] = (
    df['rel_level']
    .str.replace('c', 'Congruent')
    .str.replace('i', 'Incongruent')
)
df = df.drop(columns=['rel_level'])

df['Noise Level'] = df['noise_level']
df = df.drop(columns=['noise_level'])

df['Noise Area'] = (
    df.apply(
        lambda row: 'None' if row['Noise Level'] == 0 
        else row['Condition'].split('_')[0],
        axis=1
    )
    .replace({'target': 'Target', 'context': 'Context', 'all': 'All', 'none': 'None'})
)
df = df.drop(columns=['Condition'])

# Import data for modal names and target descriptions
modal_names_data_path = "/home/fmerlo/data/sceneregstorage/VISIONS_dataset/S3_name_agreement_critical_object.csv"
norms_numerical_data_path = "/home/fmerlo/data/sceneregstorage/VISIONS_dataset/S5_norms_numerical.csv"


modal_names_df = pd.read_csv(modal_names_data_path)
norms_numerical_df = pd.read_csv(norms_numerical_data_path)

df_modal_names_norms = modal_names_df[['sceneID', 'targetID', 'consistency', 'modalname']]
# Add one column from df2 to df1 by matching 'id'
merged_df = df_modal_names_norms.merge(norms_numerical_df[['targetID', 'object.consistency']], on='targetID', how='left')


df['targetID'] =  (
    df['image_name']
    .str.split('_')
    .str[-1] + '_' +
    df['image_name'].str.split('_').str[0] + '_' +
    df['image_name'].str.split('_').str[1]
).str.lower()
# Merge into df; use left join to keep all rows of df
df = df.merge(
    merged_df[['targetID', 'modalname', 'object.consistency']],
    on='targetID',
    how='left'
)
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
    text_embed_2 = compute_text_embedding(add_prefix(row.modalname))
    return torch.nn.functional.cosine_similarity(text_embed_1, text_embed_2, dim=1).item()

#%%
df['output_modal_name_text_similarity_scores'] = [compute_cosine_similarity(row) for row in tqdm(df.itertuples(index=False), total=len(df))]
#%%
df
#%%
# Save the updated DataFrame to a new CSV file
output_path = '/home/fmerlo/data/sceneregstorage/attn_eval_output/results_att_deployment_VISIONS_complete.csv'
df.to_csv(output_path, index=False)
print(f"Saved updated dataset to: {output_path}")