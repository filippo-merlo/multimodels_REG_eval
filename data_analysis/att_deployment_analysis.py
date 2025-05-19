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
file_path = '/home/fmerlo/data/sceneregstorage/attn_eval_output/results_att_deployment_last.csv'  # Update with your actual file path
#file_path = '/Users/filippomerlo/Desktop/attention_deployment/results_att_deployment.csv'  # Update with your actual file path

df = pd.read_csv(file_path)
df['rel_level'] = df['rel_level'].fillna('original')
df['condition'] = df['condition'].str.replace('_noise', '', regex=True)
#%%
# --- Filter dataset based on available image filenames ---
filtered_images_folder_path = '/Users/filippomerlo/Desktop/manually_filtered_images'

# Get all image filenames in the folder (only valid image formats)
image_filenames = {f for f in os.listdir(filtered_images_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))}

# Extract unique image IDs from filenames
image_filenames_id = {f.split('_')[0] for f in image_filenames}

# Ensure 'image_name' column exists before filtering
df = df[df['image_name'].apply(lambda x: x.split('_')[0] in image_filenames_id)] if 'image_name' in df.columns else df

#%%
# --- Ensure list-like columns are properly parsed from strings ---
def parse_list(value):
    """Convert string representations of lists into actual lists."""
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)  # Convert to list
        except (SyntaxError, ValueError):
            return []  # Return empty list if parsing fails
    return value  # Return value if already a list

# --- Compute attention ratio per layer ---
def compute_ratio(row):
    """Compute the ratio of attention over target to attention over context per layer."""
    attn_over_target = np.array(row['attn_over_target'], dtype=np.float32)
    attn_over_context = np.array(row['attn_over_context'], dtype=np.float32)
    
    # Avoid division by zero by using np.divide with `where` condition
    ratio = np.divide(attn_over_target, attn_over_context, out=np.full_like(attn_over_target, np.nan), where=attn_over_context!=0)
    return ratio.tolist()

df['attn_over_target'] = df['attn_over_target'].apply(parse_list)
df['attn_over_context'] = df['attn_over_context'].apply(parse_list)
df['attn_ratio'] = df.progress_apply(compute_ratio, axis=1)

# --- Expand data to have separate rows per layer ---
df['layer'] = df['attn_ratio'].apply(lambda x: list(range(len(x))))  # Add index for each layer
df_exploded = df.explode(['attn_ratio', 'layer'])  # Expand lists into rows
df_exploded['attn_ratio'] = df_exploded['attn_ratio'].apply(pd.to_numeric, errors='coerce')
#%%
# clean output and compute accuracy 
df_exploded['output_clean'] = df_exploded['output_text'].str.replace(r'<\|im_end\|>', '', regex=True).str.replace(r'\.', '', regex=True).str.lower()
df_exploded['target_clean'] = df_exploded['target'].str.replace(r' \([^)]*\)', '', regex=True).str.lower()

from Levenshtein import ratio
# Compute similarity ratio between long_output and long_target
df_exploded['Levenshtein ratio'] = df_exploded.apply(lambda row: ratio(row['output_clean'].lower(), row['target_clean'].lower()), axis=1)
df_exploded['hard_accuracy'] = df_exploded.apply(lambda row: ratio(row['output_clean'].lower(), row['target_clean'].lower()) >= 0.55, axis=1).astype(int)
#%%
# --- Filter for accuracy ---
df_exploded_correct = df_exploded[df_exploded['hard_accuracy'] == 1]
df_exploded_wrong = df_exploded[df_exploded['hard_accuracy'] == 0]

print(df_exploded.shape[0])
print(df_exploded_correct.shape[0])
print(df_exploded_wrong.shape[0])

grouped_means = df_exploded.groupby(['rel_level', 'noise_level', 'condition', 'layer'])['attn_ratio'].mean().reset_index()
grouped_means_correct = df_exploded_correct.groupby(['rel_level', 'noise_level', 'condition', 'layer'])['attn_ratio'].mean().reset_index()
grouped_means_wrong = df_exploded_wrong.groupby(['rel_level', 'noise_level', 'condition', 'layer'])['attn_ratio'].mean().reset_index()

grouped_layers = df_exploded.groupby(['rel_level', 'noise_level', 'condition'])['attn_ratio'].mean().reset_index()
grouped_layers_correct = df_exploded_correct.groupby(['rel_level', 'noise_level', 'condition'])['attn_ratio'].mean().reset_index()
grouped_layers_wrong = df_exploded_wrong.groupby(['rel_level', 'noise_level', 'condition'])['attn_ratio'].mean().reset_index()

# Merge the datasets
merged_layers = grouped_layers.merge(
    grouped_layers_correct, 
    on=['noise_level', 'rel_level', 'condition'], 
    suffixes=('_all', '_correct'),
    how='outer'  # Ensures all data is included
).merge(
    grouped_layers_wrong, 
    on=['noise_level', 'rel_level', 'condition'], 
    suffixes=('_correct', '_wrong'),
    how='outer'
)

# Rename columns explicitly to avoid naming issues
merged_layers.rename(columns={'attn_ratio': 'attn_ratio_wrong'}, inplace=True)


#%%
merged_layers.round(3)
#%%
# --- Compute mean attention ratio per layer grouped by condition ---
grouped_means = grouped_means_correct
y_lim = 1

# --- Filter and plot results for a specific noise level ---
noise_level_filter = 0.0  # Set noise level for filtering
filtered_data = grouped_means[grouped_means['noise_level'] == noise_level_filter]

plt.figure(figsize=(8, 6))
for (condition, rel_level), sub_df in filtered_data.groupby(['condition', 'rel_level']):
    if condition != 'target':
        continue
    sub_df = sub_df.sort_values(by='layer')
    plt.plot(sub_df['layer'], sub_df['attn_ratio'], marker='o', linestyle='-', label=f'Rel: {rel_level}')

plt.xlabel('Layer')
plt.ylabel('Mean Attention Ratio')
plt.ylim(0, y_lim)
plt.title(f'Mean Attention Ratio per Layer (Noise = {noise_level_filter})')
plt.legend()
plt.grid()
plt.show()

noise_level_filter = 0.5  # Set noise level for filtering
filtered_data = grouped_means[grouped_means['noise_level'] == noise_level_filter]

plt.figure(figsize=(8, 6))
for (condition, rel_level), sub_df in filtered_data.groupby(['condition', 'rel_level']):
    sub_df = sub_df.sort_values(by='layer')
    plt.plot(sub_df['layer'], sub_df['attn_ratio'], marker='o', linestyle='-', label=f'Cond: {condition}, Rel: {rel_level}')

plt.xlabel('Layer')
plt.ylabel('Mean Attention Ratio')
plt.ylim(0, y_lim)
plt.title(f'Mean Attention Ratio per Layer (Noise = {noise_level_filter})')
plt.legend()
plt.grid()
plt.show()

noise_level_filter = 1.0  # Set noise level for filtering
filtered_data = grouped_means[grouped_means['noise_level'] == noise_level_filter]

plt.figure(figsize=(8, 6))
for (condition, rel_level), sub_df in filtered_data.groupby(['condition', 'rel_level']):
    sub_df = sub_df.sort_values(by='layer')
    plt.plot(sub_df['layer'], sub_df['attn_ratio'], marker='o', linestyle='-', label=f'Cond: {condition}, Rel: {rel_level}')

plt.xlabel('Layer')
plt.ylabel('Mean Attention Ratio')
plt.ylim(0, y_lim)
plt.title(f'Mean Attention Ratio per Layer (Noise = {noise_level_filter})')
plt.legend()
plt.grid()
plt.show()

#%%
import matplotlib.pyplot as plt
import seaborn as sns

# assume grouped_means is your DataFrame
noise_levels = [0.0, 0.5, 1.0]
conditions = ['all', 'context', 'target']
n_rows, n_cols = len(noise_levels), len(conditions)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), sharex=True, sharey=True)

for i, nl in enumerate(noise_levels):
    df_n = grouped_means[grouped_means['noise_level'] == nl]
    for j, cond in enumerate(conditions):
        ax = axes[i, j]
        df_nc = df_n[df_n['condition'] == cond]
        pivot = df_nc.pivot(index='rel_level', columns='layer', values='attn_ratio')
        
        sns.heatmap(
            pivot,
            ax=ax,
            annot=False,
            fmt=".2f",
            vmin=0, vmax=1,
            linewidths=0.5,
            linecolor='gray',
            cbar=(j == n_cols - 1)
        )
        if i == 0:
            ax.set_title(cond.capitalize())
        if j == 0:
            ax.set_ylabel(f'Noise={nl}\nRelevance')
        else:
            ax.set_ylabel('')
        ax.set_xlabel('Layer')

plt.tight_layout()
plt.show()

#%%
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
# precompute global stats to center the diverging map
all_vals = grouped_means['attn_ratio']
vmin, vmax = [0,1]
vcenter = all_vals.mean()
noise_levels = [0.0, 0.5, 1.0]
conditions   = ['all', 'context', 'target']
n_rows, n_cols = len(noise_levels), len(conditions)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows),
                         sharex=True, sharey=True)

for i, nl in enumerate(noise_levels):
    df_n = grouped_means[grouped_means['noise_level'] == nl]
    for j, cond in enumerate(conditions):
        ax = axes[i, j]
        df_nc = df_n[df_n['condition'] == cond]
        pivot = df_nc.pivot(index='rel_level', columns='layer', values='attn_ratio')

        # choose either a diverging norm...
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

        sns.heatmap(
            pivot,
            ax=ax,
            annot=False,
            cmap="RdBu_r",      # diverging
            norm=norm,
            linewidths=0.5,
            linecolor='gray',
            cbar=(j == n_cols - 1)
        )

        # …or for a sequential emphasis on small ranges, swap to:
        # sns.heatmap(..., cmap="magma", vmin=vmin, vmax=vmax, ...)
        if i == 0:
            ax.set_title(cond.capitalize(), fontsize=14)
        if j == 0:
            ax.set_ylabel(f'Noise={nl}\nRelevance', fontsize=12)
        else:
            ax.set_ylabel('')
        ax.set_xlabel('Layer', fontsize=12)
        ax.tick_params(labelsize=10)

plt.tight_layout()
plt.show()


#%%
# 
r_mean_n_00_c_target_rel_original = grouped_means[(grouped_means['noise_level'] == 0.0) & (grouped_means['condition'] == 'target') & (grouped_means['rel_level'] == 'original')].sort_values(by='layer')['attn_ratio'].mean()
r_mean_n_05_c_target_rel_original = grouped_means[(grouped_means['noise_level'] == 0.5) & (grouped_means['condition'] == 'target') & (grouped_means['rel_level'] == 'original')].sort_values(by='layer')['attn_ratio'].mean()
r_mean_n_10_c_target_rel_original = grouped_means[(grouped_means['noise_level'] == 1.0) & (grouped_means['condition'] == 'target') & (grouped_means['rel_level'] == 'original')].sort_values(by='layer')['attn_ratio'].mean()
r_mean_n_00_c_context_rel_original = grouped_means[(grouped_means['noise_level'] == 0.0) & (grouped_means['condition'] == 'context') & (grouped_means['rel_level'] == 'original')].sort_values(by='layer')['attn_ratio'].mean()
r_mean_n_05_c_context_rel_original = grouped_means[(grouped_means['noise_level'] == 0.5) & (grouped_means['condition'] == 'context') & (grouped_means['rel_level'] == 'original')].sort_values(by='layer')['attn_ratio'].mean()
r_mean_n_10_c_context_rel_original = grouped_means[(grouped_means['noise_level'] == 1.0) & (grouped_means['condition'] == 'context') & (grouped_means['rel_level'] == 'original')].sort_values(by='layer')['attn_ratio'].mean()
r_mean_n_00_c_all_rel_original = grouped_means[(grouped_means['noise_level'] == 0.0) & (grouped_means['condition'] == 'all') & (grouped_means['rel_level'] == 'original')].sort_values(by='layer')['attn_ratio'].mean()
r_mean_n_05_c_all_rel_original = grouped_means[(grouped_means['noise_level'] == 0.5) & (grouped_means['condition'] == 'all') & (grouped_means['rel_level'] == 'original')].sort_values(by='layer')['attn_ratio'].mean()
r_mean_n_10_c_all_rel_original = grouped_means[(grouped_means['noise_level'] == 1.0) & (grouped_means['condition'] == 'all') & (grouped_means['rel_level'] == 'original')].sort_values(by='layer')['attn_ratio'].mean()

r_mean_n_00_c_target_rel_middle = grouped_means[(grouped_means['noise_level'] == 0.0) & (grouped_means['condition'] == 'target') & (grouped_means['rel_level'] == 'middle')].sort_values(by='layer')['attn_ratio'].mean()
r_mean_n_05_c_target_rel_middle = grouped_means[(grouped_means['noise_level'] == 0.5) & (grouped_means['condition'] == 'target') & (grouped_means['rel_level'] == 'middle')].sort_values(by='layer')['attn_ratio'].mean()
r_mean_n_10_c_target_rel_middle = grouped_means[(grouped_means['noise_level'] == 1.0) & (grouped_means['condition'] == 'target') & (grouped_means['rel_level'] == 'middle')].sort_values(by='layer')['attn_ratio'].mean()
r_mean_n_00_c_context_rel_middle = grouped_means[(grouped_means['noise_level'] == 0.0) & (grouped_means['condition'] == 'context') & (grouped_means['rel_level'] == 'middle')].sort_values(by='layer')['attn_ratio'].mean()
r_mean_n_05_c_context_rel_middle = grouped_means[(grouped_means['noise_level'] == 0.5) & (grouped_means['condition'] == 'context') & (grouped_means['rel_level'] == 'middle')].sort_values(by='layer')['attn_ratio'].mean()
r_mean_n_10_c_context_rel_middle = grouped_means[(grouped_means['noise_level'] == 1.0) & (grouped_means['condition'] == 'context') & (grouped_means['rel_level'] == 'middle')].sort_values(by='layer')['attn_ratio'].mean()
r_mean_n_00_c_all_rel_middle = grouped_means[(grouped_means['noise_level'] == 0.0) & (grouped_means['condition'] == 'all') & (grouped_means['rel_level'] == 'middle')].sort_values(by='layer')['attn_ratio'].mean()
r_mean_n_05_c_all_rel_middle = grouped_means[(grouped_means['noise_level'] == 0.5) & (grouped_means['condition'] == 'all') & (grouped_means['rel_level'] == 'middle')].sort_values(by='layer')['attn_ratio'].mean()
r_mean_n_10_c_all_rel_middle = grouped_means[(grouped_means['noise_level'] == 1.0) & (grouped_means['condition'] == 'all') & (grouped_means['rel_level'] == 'middle')].sort_values(by='layer')['attn_ratio'].mean()

r_mean_n_00_c_target_rel_low = grouped_means[(grouped_means['noise_level'] == 0.0) & (grouped_means['condition'] == 'target') & (grouped_means['rel_level'] == 'low')].sort_values(by='layer')['attn_ratio'].mean()
r_mean_n_05_c_target_rel_low = grouped_means[(grouped_means['noise_level'] == 0.5) & (grouped_means['condition'] == 'target') & (grouped_means['rel_level'] == 'low')].sort_values(by='layer')['attn_ratio'].mean()
r_mean_n_10_c_target_rel_low = grouped_means[(grouped_means['noise_level'] == 1.0) & (grouped_means['condition'] == 'target') & (grouped_means['rel_level'] == 'low')].sort_values(by='layer')['attn_ratio'].mean()
r_mean_n_00_c_context_rel_low = grouped_means[(grouped_means['noise_level'] == 0.0) & (grouped_means['condition'] == 'context') & (grouped_means['rel_level'] == 'low')].sort_values(by='layer')['attn_ratio'].mean()
r_mean_n_05_c_context_rel_low = grouped_means[(grouped_means['noise_level'] == 0.5) & (grouped_means['condition'] == 'context') & (grouped_means['rel_level'] == 'low')].sort_values(by='layer')['attn_ratio'].mean()
r_mean_n_10_c_context_rel_low = grouped_means[(grouped_means['noise_level'] == 1.0) & (grouped_means['condition'] == 'context') & (grouped_means['rel_level'] == 'low')].sort_values(by='layer')['attn_ratio'].mean()
r_mean_n_00_c_all_rel_low = grouped_means[(grouped_means['noise_level'] == 0.0) & (grouped_means['condition'] == 'all') & (grouped_means['rel_level'] == 'low')].sort_values(by='layer')['attn_ratio'].mean()
r_mean_n_05_c_all_rel_low = grouped_means[(grouped_means['noise_level'] == 0.5) & (grouped_means['condition'] == 'all') & (grouped_means['rel_level'] == 'low')].sort_values(by='layer')['attn_ratio'].mean()
r_mean_n_10_c_all_rel_low = grouped_means[(grouped_means['noise_level'] == 1.0) & (grouped_means['condition'] == 'all') & (grouped_means['rel_level'] == 'low')].sort_values(by='layer')['attn_ratio'].mean()


# Create a dictionary to store the results
results = {
    ("target", "original"): [r_mean_n_00_c_target_rel_original, r_mean_n_05_c_target_rel_original, r_mean_n_10_c_target_rel_original],
    ("context", "original"): [r_mean_n_00_c_context_rel_original, r_mean_n_05_c_context_rel_original, r_mean_n_10_c_context_rel_original],
    ("all", "original"): [r_mean_n_00_c_all_rel_original, r_mean_n_05_c_all_rel_original, r_mean_n_10_c_all_rel_original],
    ("target", "middle"): [r_mean_n_00_c_target_rel_middle, r_mean_n_05_c_target_rel_middle, r_mean_n_10_c_target_rel_middle],
    ("context", "middle"): [r_mean_n_00_c_context_rel_middle, r_mean_n_05_c_context_rel_middle, r_mean_n_10_c_context_rel_middle],
    ("all", "middle"): [r_mean_n_00_c_all_rel_middle, r_mean_n_05_c_all_rel_middle, r_mean_n_10_c_all_rel_middle],
    ("target", "low"): [r_mean_n_00_c_target_rel_low, r_mean_n_05_c_target_rel_low, r_mean_n_10_c_target_rel_low],
    ("context", "low"): [r_mean_n_00_c_context_rel_low, r_mean_n_05_c_context_rel_low, r_mean_n_10_c_context_rel_low],
    ("all", "low"): [r_mean_n_00_c_all_rel_low, r_mean_n_05_c_all_rel_low, r_mean_n_10_c_all_rel_low],
}

# Convert to DataFrame
df_results = pd.DataFrame.from_dict(results, orient="index", columns=["Noise 0.0", "Noise 0.5", "Noise 1.0"])

# Rename index for better readability
df_results.index = pd.MultiIndex.from_tuples(df_results.index, names=["Condition", "Relatedness Level"])

# Replace NaN values with "Not Available"
df_results = df_results.fillna("Not Available")

# Print the table
print(df_results)