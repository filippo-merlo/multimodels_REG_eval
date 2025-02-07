#%%
import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

tqdm.pandas()

separator = "\n\n##################################################\n##################################################\n\n"

# --- Load the data ---
file_path = '/Users/filippomerlo/Desktop/attention_deployment/results_att_deployment.csv'  # Update with your actual file path
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

#%%
# --- Compute mean attention ratio per layer grouped by condition ---
grouped_means = df_exploded.groupby(['condition', 'noise_level', 'rel_level', 'layer'])['attn_ratio'].mean().reset_index()

#%%
# --- Filter and plot results for a specific noise level ---
noyse_level_filter = 0.0  # Set noise level for filtering
filtered_data = grouped_means[grouped_means['noise_level'] == noyse_level_filter]

plt.figure(figsize=(8, 6))
for (condition, rel_level), sub_df in filtered_data.groupby(['condition', 'rel_level']):
    if condition != 'target':
        continue
    sub_df = sub_df.sort_values(by='layer')
    plt.plot(sub_df['layer'], sub_df['attn_ratio'], marker='o', linestyle='-', label=f'Rel: {rel_level}')

plt.xlabel('Layer')
plt.ylabel('Mean Attention Ratio')
plt.title(f'Mean Attention Ratio per Layer (Noise = {noyse_level_filter})')
plt.legend()
plt.grid()
plt.show()

noyse_level_filter = 0.5  # Set noise level for filtering
filtered_data = grouped_means[grouped_means['noise_level'] == noyse_level_filter]

plt.figure(figsize=(8, 6))
for (condition, rel_level), sub_df in filtered_data.groupby(['condition', 'rel_level']):
    sub_df = sub_df.sort_values(by='layer')
    plt.plot(sub_df['layer'], sub_df['attn_ratio'], marker='o', linestyle='-', label=f'Cond: {condition}, Rel: {rel_level}')

plt.xlabel('Layer')
plt.ylabel('Mean Attention Ratio')
plt.title(f'Mean Attention Ratio per Layer (Noise = {noyse_level_filter})')
plt.legend()
plt.grid()
plt.show()

noyse_level_filter = 1.0  # Set noise level for filtering
filtered_data = grouped_means[grouped_means['noise_level'] == noyse_level_filter]

plt.figure(figsize=(8, 6))
for (condition, rel_level), sub_df in filtered_data.groupby(['condition', 'rel_level']):
    sub_df = sub_df.sort_values(by='layer')
    plt.plot(sub_df['layer'], sub_df['attn_ratio'], marker='o', linestyle='-', label=f'Cond: {condition}, Rel: {rel_level}')

plt.xlabel('Layer')
plt.ylabel('Mean Attention Ratio')
plt.title(f'Mean Attention Ratio per Layer (Noise = {noyse_level_filter})')
plt.legend()
plt.grid()
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