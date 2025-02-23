#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from pprint import pprint

# Specify the folder containing your CSV files
folder_path = '/Users/filippomerlo/Desktop/output'

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

desired_models = [
    'cyan2k/molmo-7B-O-bnb-4bit',
    'Salesforce/xgen-mm-phi3-mini-instruct-r-v1',
    'llava-hf/llava-onevision-qwen2-0.5b-si-hf',
    'Salesforce/xgen-mm-phi3-mini-instruct-singleimg-r-v1.5',
    'Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8',
    'microsoft/kosmos-2-patch14-224'
]

# Filter DataFrame to include only the selected models
df = df[df['model_name'].isin(desired_models)]
df
#%%
# --- Filter dataset based on selected image ---
filtered_images_folder_path = '/Users/filippomerlo/Desktop/manually_filtered_images'
# Get all image filenames in the folder (only valid image formats)
image_filenames = {f for f in os.listdir(filtered_images_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))}
# Extract unique image IDs from filenames
image_filenames_id = {f.split('_')[0] for f in image_filenames}
# filter by image filenames
df_filtered = df[df['image_name'].apply(lambda x: x.split('_')[0] in image_filenames_id)] if 'image_name' in df.columns else df


# compute metrics
# Full Dataset
# Hard Accuracy
from Levenshtein import ratio
# Compute similarity ratio between long_output and long_target
df['target_clean'] = df['target'].str.replace(r' \([^)]*\)', '', regex=True).str.lower()
df['output_clean'] = df['output'].str.replace('\.', '', regex=True).str.lower()
df['Levenshtein ratio'] = df.apply(lambda row: ratio(row['output_clean'], row['target_clean']), axis=1)
df['hard_accuracy'] = df.apply(lambda row: row['Levenshtein ratio'] >= 0.55, axis=1).astype(int)
# Find the longest string in 'target_clean'
max_length = max(df['target_clean'].apply(len))
print(max_length)
# Filter out rows where 'output_clean' is longer than max_length
df_hard_accuracy = df[df['output_clean'].apply(len) <= max_length] #!!!
hard_accuracy_by_combined = df_hard_accuracy.groupby(['Rel. Level', 'noise_level', 'Noise Area'])[
    ['hard_accuracy']
].mean().reset_index()

# Soft Accuracy
df['soft_accuracy'] = (df['long_caption_text_similarity_scores'] >= 0.9).astype(int) #!!!
soft_accuracy_by_combined = df.groupby(['Rel. Level', 'noise_level', 'Noise Area'])[
    ['soft_accuracy']
].mean().reset_index()

# Filtered Dataset
# Hard Accuracy
df_filtered['target_clean'] = df_filtered['target'].str.replace(r' \([^)]*\)', '', regex=True).str.lower()
df_filtered['output_clean'] = df_filtered['output'].str.replace('\.', '', regex=True).str.lower()
df_filtered['Levenshtein ratio'] = df_filtered.apply(lambda row: ratio(row['output_clean'], row['target_clean']), axis=1)
df_filtered['hard_accuracy'] = df_filtered.apply(lambda row: row['Levenshtein ratio'] >= 0.55, axis=1).astype(int)
# Find the longest string in 'target_clean'
max_length = max(df_filtered['target_clean'].apply(len))
# Filter out rows where 'output_clean' is longer than max_length
df_filtered_hard_accuracy = df_filtered[df_filtered['output_clean'].apply(len) <= max_length] #!!!
hard_accuracy_filtered_by_combined = df_filtered_hard_accuracy.groupby(['Rel. Level', 'noise_level', 'Noise Area'])[
    ['hard_accuracy']
].mean().reset_index()

# Soft Accuracy
df_filtered['soft_accuracy'] = (df_filtered['long_caption_text_similarity_scores'] >= 0.9).astype(int) #!!!
soft_accuracy_filtered_by_combined = df_filtered.groupby(['Rel. Level', 'noise_level', 'Noise Area'])[
    ['soft_accuracy']
].mean().reset_index()

#%%

# Merge datasets, hard accuracy
merged_hard_accuracy = hard_accuracy_by_combined.merge(
    hard_accuracy_filtered_by_combined, 
    on=['Rel. Level', 'noise_level', 'Noise Area'], 
    suffixes=('_full', '_filtered'),
    how='outer'  # Ensures all data is included
).round(3)

# Merge datasets, soft accuracy
merged_soft_accuracy = soft_accuracy_by_combined.merge(
    soft_accuracy_filtered_by_combined, 
    on=['Rel. Level', 'noise_level', 'Noise Area'], 
    suffixes=('_full', '_filtered'),
    how='outer'  # Ensures all data is included
).round(3)

merged_soft_accuracy
#%%
# Group data by 'rel_level', 'noise_level', and 'condition', then compute mean scores
semantic_by_combined = df.groupby(['Rel. Level', 'noise_level', 'Noise Area'])[
    ['long_caption_scores', 'long_caption_text_similarity_scores']
].mean().reset_index()

# Group data by 'rel_level', 'noise_level', and 'condition', then compute mean scores
semantic_by_combined_filtered = df_filtered.groupby(['Rel. Level', 'noise_level', 'Noise Area'])[
    ['long_caption_scores', 'long_caption_text_similarity_scores']
].mean().reset_index()

# Merge datasets, keeping track of filtered data
merged_semantic = semantic_by_combined.merge(
    semantic_by_combined_filtered, 
    on=['Rel. Level', 'noise_level', 'Noise Area'], 
    suffixes=('_full', '_filtered'),
    how='outer'  # Ensures all data is included
).round(3)

#%% VISUALIZE
# Filter dataset to only include rows with noise_level == 0
df_zero_noise = df[df['noise_level'] == 0]

# Group data by 'rel_level' and 'condition', then compute mean scores
performance_by_zero_noise = df_zero_noise.groupby(['Rel. Level', 'Noise Area', 'model_name'])[
    ['scores', 'text_similarity_scores', 'long_caption_scores', 'long_caption_text_similarity_scores']
].median().reset_index()

# List of scores to visualize
scores = ['long_caption_scores']

# Generate bar plots for each score metric at noise level 0
for score in scores:
    plt.figure(figsize=(14, 7))
    sns.barplot(
        data=performance_by_zero_noise,
        x='Rel. Level',
        y=score,
        hue='model_name',
        palette='tab10'
    )
    plt.ylim(0.5,0.8)
    plt.title(f'RefCLIPScore at Noise Level 0 by Relatedness Level')
    plt.xlabel('Relatedness Level')
    plt.ylabel('RefCLIPScore')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

#%% Semantic Similarity Analysis

# List of scores to visualize
scores = ['long_caption_scores']

# Generate line plots for each score metric
for score in scores:
    plt.figure(figsize=(14, 7))
    sns.lineplot(
        data=semantic_by_combined,
        x='noise_level',
        y=score,
        hue='Rel. Level',
        style='Noise Area',  # Differentiates between conditions
        markers=True,
        palette='tab10'
    )
    plt.title(f'RefCLIPScore vs Noise Level by Noise Area and Relatedness Level')
    plt.xlabel('Noise Level')
    plt.xticks([0,0.5,1])
    plt.ylabel('RefCLIPScore')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

#%%
pivot_table = semantic_by_combined.pivot_table(
    index="Rel. Level", columns=["noise_level", "Noise Area"], values="long_caption_scores"
)

plt.figure(figsize=(12, 6))
sns.heatmap(pivot_table, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Heatmap of Scores by Noise Level, Condition, and Rel Level")
plt.show()

#%% Accuracy Analysis

# List of accuracy metrics to visualize
accuracies = ['hard_accuracy']

# Generate line plots for accuracy metrics
for accuracy in accuracies:
    plt.figure(figsize=(14, 7))
    sns.lineplot(
        data=hard_accuracy_by_combined,
        x='noise_level',
        y=accuracy,
        hue='Rel. Level',
        style='Noise Area',  # Differentiates between conditions
        markers=True,
        palette='tab10'
    )
    plt.title(f'{accuracy.replace("_", " ").capitalize()} vs Noise Level by Noise Area and Rel Level')
    plt.xlabel('Noise Level')
    plt.xticks([0,0.5,1])
    plt.ylabel(f'{accuracy.replace("_", " ").capitalize()}')
    plt.legend(title='Noise Area / Rel Level', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
