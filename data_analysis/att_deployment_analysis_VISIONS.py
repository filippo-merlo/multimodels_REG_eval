#%%
import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import os
from pprint import pprint

tqdm.pandas()

plt.rcParams.update({
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.title_fontsize': 14,
    'legend.fontsize': 12,
    'font.size': 14  # base font size
})

separator = "\n\n##################################################\n##################################################\n\n"

# --- Load the data ---
dataset_path = '/home/fmerlo/data/sceneregstorage/attn_eval_output/results_att_deployment_VISIONS.csv'  # Update with your actual file path

df = pd.read_csv(dataset_path)
for col in df.columns:
    print(f"\nColumn: {col}")
    print(df[col].unique())

#%%
# Preprocess

df['condition'] = df['condition'].str.replace('_noise', '', regex=True)
df['output_clean'] = df['output_text'].str.replace(r'<\|im_end\|>', '', regex=True).str.replace(r'\.', '', regex=True).str.lower()

df['Rel. Level'] = df['rel_level']
df = df.drop(columns=['rel_level'])
df['Rel. Level'] = df['Rel. Level'].apply(lambda x: x.replace('c', 'Congruent')).apply(lambda x: x.replace('i', 'Incongruent'))
df['condition'] = df['condition'].str.replace('_noise', '', regex=True)
df['Noise Area'] = df['condition'].apply(lambda x: x.split('_')[0]).apply(lambda x: x.replace('target', 'Target')).apply(lambda x: x.replace('context', 'Context')).apply(lambda x: x.replace('all', 'All')).apply(lambda x: x.replace('none', 'None'))
df = df.drop(columns=['condition'])
df['Noise Level'] = df['noise_level']
df = df.drop(columns=['noise_level'])
#%%
for col in df.columns:
    print(f"\nColumn: {col}")
    print(df[col].unique())

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
# compute soft accuracy
df_exploded['soft_accuracy'] = (df_exploded['long_caption_text_similarity_score'] >= 0.9).astype(int) #!!!

# clean output and compute hard accuracy 
df_exploded['output_clean'] = df_exploded['output_text'].str.replace(r'<\|im_end\|>', '', regex=True).str.replace(r'\.', '', regex=True).str.lower()
df_exploded['target_clean'] = df_exploded['target'].str.replace(r' \([^)]*\)', '', regex=True).str.lower()

from Levenshtein import ratio
# Compute similarity ratio between long_output and long_target
df_exploded['Levenshtein ratio'] = df_exploded.apply(lambda row: ratio(row['output_clean'].lower(), row['target_clean'].lower()), axis=1)
df_exploded['hard_accuracy'] = df_exploded.apply(lambda row: ratio(row['output_clean'].lower(), row['target_clean'].lower()) >= 0.55, axis=1).astype(int)
#%%
# --- Filter for accuracy ---
#df_exploded_correct = df_exploded[df_exploded['hard_accuracy'] == 1]
#df_exploded_wrong = df_exploded[df_exploded['hard_accuracy'] == 0]
df_exploded_correct = df_exploded[df_exploded['soft_accuracy'] == 1]
df_exploded_wrong = df_exploded[df_exploded['soft_accuracy'] == 0]


print(df_exploded.shape[0])
print(df_exploded_correct.shape[0])
print(df_exploded_wrong.shape[0])

grouped_means_complete = df_exploded.groupby(['Rel. Level', 'Noise Level', 'Noise Area', 'layer','soft_accuracy'])['attn_ratio'].mean().reset_index()
grouped_means = df_exploded.groupby(['Rel. Level', 'Noise Level', 'Noise Area', 'layer'])['attn_ratio'].mean().reset_index()
grouped_means_correct = df_exploded_correct.groupby(['Rel. Level', 'Noise Level', 'Noise Area', 'layer'])['attn_ratio'].mean().reset_index()
grouped_means_wrong = df_exploded_wrong.groupby(['Rel. Level', 'Noise Level', 'Noise Area', 'layer'])['attn_ratio'].mean().reset_index()

grouped_layers = df_exploded.groupby(['Rel. Level', 'Noise Level', 'Noise Area'])['attn_ratio'].mean().reset_index()
grouped_layers_correct = df_exploded_correct.groupby(['Rel. Level', 'Noise Level', 'Noise Area'])['attn_ratio'].mean().reset_index()
grouped_layers_wrong = df_exploded_wrong.groupby(['Rel. Level', 'Noise Level', 'Noise Area'])['attn_ratio'].mean().reset_index()

# Merge the datasets
merged_layers = grouped_layers.merge(
    grouped_layers_correct, 
    on=['Noise Level', 'Rel. Level', 'Noise Area'], 
    suffixes=('_all', '_correct'),
    how='outer'  # Ensures all data is included
).merge(
    grouped_layers_wrong, 
    on=['Noise Level', 'Rel. Level', 'Noise Area'], 
    suffixes=('_correct', '_wrong'),
    how='outer'
)

# Rename columns explicitly to avoid naming issues
merged_layers.rename(columns={'attn_ratio': 'attn_ratio_wrong'}, inplace=True)
#%%
print(merged_layers.round(3).to_latex(index=False))
#%%
# --- Compute mean attention ratio per layer grouped by condition ---
grouped_means = grouped_means_wrong
y_lim = 1

# --- Filter and plot results for a specific noise level ---
noise_level_filter = 0.0  # Set noise level for filtering
filtered_data = grouped_means[grouped_means['Noise Level'] == noise_level_filter]

plt.figure(figsize=(8, 6))
for (condition, rel_level), sub_df in filtered_data.groupby(['Noise Area', 'Rel. Level']):
    if condition != 'Target':
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
filtered_data = grouped_means[grouped_means['Noise Level'] == noise_level_filter]

plt.figure(figsize=(8, 6))
for (condition, rel_level), sub_df in filtered_data.groupby(['Noise Area', 'Rel. Level']):
    sub_df = sub_df.sort_values(by='layer')
    plt.plot(sub_df['layer'], sub_df['attn_ratio'], marker='o', linestyle='-', label=f'Area: {condition}, Rel: {rel_level}')

plt.xlabel('Layer')
plt.ylabel('Mean Attention Ratio')
plt.ylim(0, y_lim)
plt.title(f'Mean Attention Ratio per Layer (Noise = {noise_level_filter})')
plt.legend()
plt.grid()
plt.show()

noise_level_filter = 1.0  # Set noise level for filtering
filtered_data = grouped_means[grouped_means['Noise Level'] == noise_level_filter]

plt.figure(figsize=(8, 6))
for (condition, rel_level), sub_df in filtered_data.groupby(['Noise Area', 'Rel. Level']):
    sub_df = sub_df.sort_values(by='layer')
    plt.plot(sub_df['layer'], sub_df['attn_ratio'], marker='o', linestyle='-', label=f'Area: {condition}, Rel: {rel_level}')

plt.xlabel('Layer')
plt.ylabel('Mean Attention Ratio')
plt.ylim(0, y_lim)
plt.title(f'Mean Attention Ratio per Layer (Noise = {noise_level_filter})')
plt.legend()
plt.grid()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# assume grouped_means is your DataFrame
noise_levels = [0.0, 0.5, 1.0]
conditions = ['all', 'context', 'target']
n_rows, n_cols = len(noise_levels), len(conditions)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), sharex=True, sharey=True)

for i, nl in enumerate(noise_levels):
    df_n = grouped_means[grouped_means['Noise Level'] == nl]
    for j, cond in enumerate(conditions):
        ax = axes[i, j]
        df_nc = df_n[df_n['Noise Area'] == cond]
        pivot = df_nc.pivot(index='Rel. Level', columns='layer', values='attn_ratio')
        
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

from matplotlib.colors import TwoSlopeNorm
import matplotlib.gridspec as gridspec

# precompute global stats to center the diverging map
all_vals = grouped_means['attn_ratio']
vmin, vmax = [0, 1]
vcenter = all_vals.mean()
vcenter = grouped_means[grouped_means['Noise Level'] == 0.0]['attn_ratio'].mean()
#vcenter = 0.18 # avg of 0 noise all 
print(f"vmin: {vmin}, vcenter: {vcenter}, vmax: {vmax}")
noise_levels = [0.0, 0.5, 1.0]
conditions = ['all', 'context', 'target']
n_rows, n_cols = len(noise_levels), len(conditions)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=True, sharey=True)

for i, nl in enumerate(noise_levels):
    df_n = grouped_means[grouped_means['Noise Level'] == nl]
    for j, cond in enumerate(conditions):
        ax = axes[i, j]
        df_nc = df_n[df_n['Noise Area'] == cond]
        pivot = df_nc.pivot(index='Rel. Level', columns='layer', values='attn_ratio')

        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

        sns.heatmap(
            pivot,
            ax=ax,
            annot=False,
            cmap="RdBu_r",
            norm=norm,
            linewidths=0.5,
            linecolor='gray',
            cbar=(j == n_cols - 1)
        )
        if j == n_cols - 1:
            cbar = ax.collections[0].colorbar
            raw_ticks = np.arange(vmin, vmax + 0.01, 0.2)
            ticks = [round(t, 2) for t in raw_ticks if abs(t - vcenter) >= 0.05]
            ticks.append(round(vcenter, 2))
            ticks = sorted(set(ticks))
            cbar.set_ticks(ticks)
            cbar.ax.set_yticklabels([
                f"$\\bf{{{t:.2f}}}$" if np.isclose(t, vcenter) else f"{t:.2f}"
                for t in ticks
            ])


        if i == 0:
            ax.set_title(cond.capitalize(), fontsize=20)
        if j == 0:
            ax.set_ylabel(f'Noise={nl}\nRelevance', fontsize=16)
        else:
            ax.set_ylabel('')
        ax.tick_params(axis='x', rotation=0)
        ax.set_xlabel('Layer', fontsize=16)
        ax.tick_params(labelsize=14)

plt.subplots_adjust(right=0.90)
plt.tight_layout()
plt.show()


#%%
from matplotlib.colors import TwoSlopeNorm
# ====== DELTAS ========
# Merge on all relevant grouping keys
merged = pd.merge(
    grouped_means_correct,
    grouped_means_wrong,
    on=['Rel. Level', 'Noise Level', 'Noise Area', 'layer'],
    suffixes=('_correct', '_wrong')
)

# Compute the delta
merged['attn_ratio_delta'] = merged['attn_ratio_correct'] - merged['attn_ratio_wrong']
merged['abs_delta'] = merged['attn_ratio_delta'].abs()
top_deltas = merged.sort_values(by='abs_delta', ascending=False).head(10)
top_deltas
#%%
vmin, vmax = -0.25, 0.25  # adjust depending on your actual deltas
vcenter = 0.0  # because we're plotting difference
print(f"vmin: {vmin}, vcenter: {vcenter}, vmax: {vmax}")

noise_levels = [0.0, 0.5, 1.0]
conditions = ['all', 'context', 'target']
n_rows, n_cols = len(noise_levels), len(conditions)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=True, sharey=True)

for i, nl in enumerate(noise_levels):
    df_n = merged[merged['Noise Level'] == nl]
    for j, cond in enumerate(conditions):
        ax = axes[i, j]
        df_nc = df_n[df_n['Noise Area'] == cond]
        pivot = df_nc.pivot(index='Rel. Level', columns='layer', values='attn_ratio_delta')

        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

        sns.heatmap(
            pivot,
            ax=ax,
            annot=False,
            cmap="PiYG",
            norm=norm,
            linewidths=0.5,
            linecolor='gray',
            cbar=(j == n_cols - 1)
        )
        if j == n_cols - 1:
            cbar = ax.collections[0].colorbar
            raw_ticks = np.linspace(vmin, vmax, 11)
            ticks = [round(t, 2) for t in raw_ticks if abs(t - vcenter) >= 0.05]
            ticks.append(round(vcenter, 2))
            ticks = sorted(set(ticks))
            cbar.set_ticks(ticks)
            cbar.ax.set_yticklabels([
                f"$\\bf{{{t:.2f}}}$" if np.isclose(t, vcenter) else f"{t:.2f}"
                for t in ticks
            ])

        if i == 0:
            ax.set_title(cond.capitalize(), fontsize=20)
        if j == 0:
            ax.set_ylabel(f'Noise={nl}\nRelevance', fontsize=16)
        else:
            ax.set_ylabel('')
        ax.tick_params(axis='x', rotation=0)
        ax.set_xlabel('Layer', fontsize=16)
        ax.tick_params(labelsize=14)

plt.tight_layout()
plt.show()


