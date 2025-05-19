#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from pprint import pprint
from transformers import CLIPModel, CLIPProcessor
import torch
from tqdm import tqdm
# Load CLIP model and tokenizer

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
    text_embed_1 = compute_text_embedding(row.long_output)
    text_embed_2 = compute_text_embedding(add_prefix(row.scene))
    return torch.nn.functional.cosine_similarity(text_embed_1, text_embed_2, dim=1).item()

# Specify the folder containing your CSV files
folder_path = '/home/fmerlo/data/sceneregstorage/eval_output'

# Get a list of all CSV files in the specified folder
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# Initialize an empty list to store DataFrames
dataframes = []

# Loop through each CSV file and read it into a pandas DataFrame
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    dataframes.append(df)

# Concatenate all DataFrames into one combined DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)
df = combined_df.copy()

# Filter out rows where 'target' is "nothing"
df = df[df['target'] != "nothing"]

# Fill missing values in 'rel_level' with 'original'
df['Rel. Level'] = df['rel_level'].fillna('original')
df['Noise Area'] = df['condition'].apply(lambda x: x.split('_')[0])

#%%
# Apply the function to each row with a progress bar
df['scene_output_similarity'] = [compute_cosine_similarity(row) for row in tqdm(df.itertuples(index=False), total=len(df))]
df
# Save the updated DataFrame back to CSV
df.to_csv(os.path.join(folder_path, 'updated_complete_output.csv'), index=False)
#%%
df