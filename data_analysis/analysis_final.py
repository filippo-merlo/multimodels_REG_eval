#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint

#%%
# Specify the folder containing your CSV files
file_path = '/home/fmerlo/data/sceneregstorage/eval_output/dataset_final_final_complete.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Set global font sizes
sns.set_context("talk")  # options: paper, notebook, talk, poster
plt.rcParams.update({
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.title_fontsize': 14,
    'legend.fontsize': 12,
})

set(df['model_name'])

#%%
for i, (_, row) in enumerate(df.iterrows()):
    if i > 100:
        break
    print('####')
    print(row['image_name'])
    print(row['original_target'])
    print(row['output_clean'])
    print(row['original_target_output_similarity'])
#%%

desired_models = [
    'Qwen/Qwen2.5-VL-7B-Instruct',
    'allenai/Molmo-7B-D-0924',
    'llava-hf/llava-onevision-qwen2-7b-ov-hf',
    'Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5',
    'llava-hf/llava-onevision-qwen2-0.5b-si-hf',
    'microsoft/kosmos-2-patch14-224'
]

# Filter DataFrame to include only the selected models
df = df[df['model_name'].isin(desired_models)]
df
#%% Preliminary statistics
# Compute the average rel_score per Rel. Level
avg_scores = df.groupby("Rel. Level")["rel_score"].mean()

print(avg_scores)
# Reset index and convert to categorical with ordered levels
avg_scores_df = avg_scores.reset_index()
rel_order = ['original', 'same target', 'high', 'medium', 'low']
avg_scores_df["Rel. Level"] = pd.Categorical(avg_scores_df["Rel. Level"], categories=rel_order, ordered=True)

# Sort values by custom order
avg_scores_df = avg_scores_df.sort_values("Rel. Level")

# Plot
sns.barplot(data=avg_scores_df, x="Rel. Level", y="rel_score", palette="Blues_d", edgecolor="black")
plt.title("Average Rel. Score by Rel. Level")
plt.xlabel("Rel. Level")
plt.ylabel("Average Rel. Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%%
# --- Filter dataset based on selected image ---
#filtered_images_folder_path = '/Users/filippomerlo/Desktop/manually_filtered_images'
## Get all image filenames in the folder (only valid image formats)
#image_filenames = {f for f in os.listdir(filtered_images_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))}
## Extract unique image IDs from filenames
#image_filenames_id = {f.split('_')[0] for f in image_filenames}
## filter by image filenames
#df_filtered = df[df['image_name'].apply(lambda x: x.split('_')[0] in image_filenames_id)] if 'image_name' in df.columns else df
#%%
# compute metrics
# Soft Accuracy
soft_accuracy_by_combined = df.groupby(['Rel. Level', 'Noise Level', 'Noise Area'])[
    ['soft_accuracy']
].mean().reset_index()

# Hard Accuracy
# Find the longest string in 'target_clean'
max_length = max(df['target_clean'].apply(len))
print(max_length)

# Filter out rows where 'output_clean' is longer than max_length
df_hard_accuracy = df[df['output_clean'].apply(len) <= max_length] #!!!
hard_accuracy_by_combined = df_hard_accuracy.groupby(['Rel. Level', 'Noise Level', 'Noise Area'])[
    ['hard_accuracy']
].mean().reset_index()

#%%

df_to_print = df_hard_accuracy.groupby(['model_name', 'Rel. Level', 'Noise Level', 'Noise Area'])[['long_caption_scores', 'long_caption_text_similarity_scores', 'hard_accuracy', 'soft_accuracy']].mean().reset_index()
df_to_print.columns = ['model_name', 'Rel. Level', 'Noise Level', 'Noise Area',
       'refCLIPScore', 'Text-Based Similarity',
       'Hard Acc.', 'Soft Acc.']

for model, group in df_to_print.groupby('model_name'):
    print('\n')
    # Optionally drop the 'model_name' column if it's redundant
    print("\\begin{table}[h]")
    print("\hspace{-1.5cm}")
    print(group.drop(columns='model_name').to_latex(index=False, float_format="%.3f"))
    print("\caption{Results for model: " + model + "}")
    print("\end{table}")

#%%
df.columns

#%%
# Hard Accuracy
df['correct_soft'] = df['soft_accuracy'] == 1
soft_accuracy_similarity = df.groupby(
    ['Noise Area', 'Noise Level', 'Rel. Level', 'correct_soft']
)[['scene_output_similarity']].mean().unstack()
#)[['original_target_output_similarity']].mean().unstack()


# Renaming columns for clarity
#hard_accuracy_similarity.columns = ['Target/Out. Incorrect', 'Target/Out. Correct', 'Scene/Out. Incorrect', 'Scene/Out. Correct']
#hard_accuracy_similarity = hard_accuracy_similarity[['Target/Out. Correct', 'Target/Out. Incorrect', 'Scene/Out. Correct', 'Scene/Out. Incorrect']]
soft_accuracy_similarity.columns = ['Incorrect', 'Correct']
soft_accuracy_similarity = soft_accuracy_similarity[['Correct', 'Incorrect']]


merged_accuracy_similarity = soft_accuracy_similarity.reset_index()
merged_accuracy_similarity = merged_accuracy_similarity[
    ~((merged_accuracy_similarity['Noise Level'] == 0.0) & 
      (merged_accuracy_similarity['Noise Area'] != 'target'))
]
merged_accuracy_similarity['Noise Area'][
    ((merged_accuracy_similarity['Noise Level'] == 0.0) & 
      (merged_accuracy_similarity['Noise Area'] == 'target'))
] = '--'

# Define the desired order
desired_order_area = ['--', 'target', 'context', 'all']
desired_order_level = [0.0, 0.5, 1.0]
desired_order_rel = ['original','same target', 'high', 'medium', 'low']

# Convert column to categorical with the specified order
merged_accuracy_similarity['Noise Area'] = pd.Categorical(merged_accuracy_similarity['Noise Area'], categories=desired_order_area, ordered=True)
merged_accuracy_similarity['Noise Level'] = pd.Categorical(merged_accuracy_similarity['Noise Level'], categories=desired_order_level, ordered=True)
merged_accuracy_similarity['Rel. Level'] = pd.Categorical(merged_accuracy_similarity['Rel. Level'], categories=desired_order_rel, ordered=True)


merged_accuracy_similarity = merged_accuracy_similarity.set_index(['Noise Area', 'Noise Level', 'Rel. Level'])
merged_accuracy_similarity.round(3)

# Reset index to have all categories as columns for easier plotting
merged_accuracy_similarity = merged_accuracy_similarity.reset_index()

#merged_accuracy_similarity = merged_accuracy_similarity[merged_accuracy_similarity['Noise Level'] != 1.0]
merged_accuracy_similarity

#%%
# Melt the dataframe for seaborn compatibility
df_melted = merged_accuracy_similarity.melt(
    id_vars=['Noise Area', 'Noise Level', 'Rel. Level'],
    var_name='Accuracy Type',
    value_name='Scene Output Similarity'
)

import matplotlib.ticker as mticker

# Plot using seaborn with noise area, noise level, and relative level
plt.figure(figsize=(14, 8))
g = sns.catplot(
    data=df_melted,
    x='Noise Level',
    y='Scene Output Similarity',
    hue='Accuracy Type',
    col='Noise Area',
    row='Rel. Level',
    kind='bar',
    height=4,
    aspect=1.2,
    #palette='rocket',
    sharex=False  # Disable shared x-axis
)

# Customize the plotm
for idx, ax in enumerate(g.axes.flat):
    #ax.set_ylim(0.77, 0.86)
    ax.set_ylim(0.77, 0.86)
    if idx in [0,4,8,12,16]:
        ax.set_xticks([0])  
        ax.margins(x=0.9)
    else:
        ax.set_xticks([1,2])
        ax.margins(x=0.1)

    # Increase font sizes for axis labels and tick labels
    ax.xaxis.label.set_size(16)  # X-axis label size
    ax.yaxis.label.set_size(16)  # Y-axis label size
    ax.tick_params(axis='both', labelsize=14)  # Tick labels size
    ax.title.set_size(14)  # Subplot title size
    # Add grid lines
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)  # Dashed grid lines with transparency

g.set_axis_labels("Noise Level", "Semantic Similarity")
plt.subplots_adjust(top=0.9)
plt.suptitle('Scene-Output Text Based Semantic Similarity', fontsize=24)

# Increase legend font size
legend = g._legend
if legend:
    plt.setp(legend.get_texts(), fontsize=18)  # Increase legend text size
    legend.set_title("", prop={'size': 20})  # Increase legend title size

    # Move the legend to the right and rotate it vertically
    legend.set_bbox_to_anchor((1.06, 0.5))  # Moves it outside the plot, center right

# Show plot
plt.show()
#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# --- Prepare data ------------------------------------------------------------
df_melted = merged_accuracy_similarity.melt(
    id_vars=['Noise Area', 'Noise Level', 'Rel. Level'],
    var_name='Accuracy Type',
    value_name='Scene Output Similarity'
)
df_melted['Noise Level'] = df_melted['Noise Level'].astype(float)

df_melted['Rel. Level Short'] = df_melted['Rel. Level'].replace({
    'original': 'original',
    'same target': 'same target',
    'high': 'high',
    'medium': 'medium',
    'low': 'low'
})

df_melted['Noise Area Short'] = df_melted['Noise Area'].replace({
    'target': 'target',
    'context': 'context',
    'all': 'all'
    # NOTE: we intentionally do NOT touch the '--' / '–' value here
})

# --- Define baseline vs. other areas (use original Noise Area) ---------------
baseline_mask = df_melted['Noise Area'].isin(['--', '–', 'none'])
baseline = df_melted[baseline_mask].copy()
df_plot = df_melted[~baseline_mask].copy()

row_order = ['original', 'same target', 'high', 'medium', 'low']
col_order = ['target', 'context', 'all']

df_plot['Rel. Level Short'] = pd.Categorical(df_plot['Rel. Level Short'],
                                             categories=row_order, ordered=True)
df_plot['Noise Area Short'] = pd.Categorical(df_plot['Noise Area Short'],
                                             categories=col_order, ordered=True)

# one baseline value per Rel × Accuracy Type
baseline_mean = (
    baseline
    .groupby(['Rel. Level Short', 'Accuracy Type'], as_index=False)
    ['Scene Output Similarity'].mean()
)

# --- Style -------------------------------------------------------------------
sns.set_theme(
    style="white",
    context="paper",
    rc={
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "axes.edgecolor": "0.2",
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
    }
)

g = sns.relplot(
    data=df_plot,
    x='Noise Level',
    y='Scene Output Similarity',
    hue='Accuracy Type',
    style='Accuracy Type',
    kind='line',
    markers=True,
    linewidth=1.5,
    markersize=5,
    col='Noise Area Short',
    row='Rel. Level Short',
    col_order=col_order,
    row_order=row_order,
    height=2.1,
    aspect=1.4,
    facet_kws={'margin_titles': True}
)

# --- color mapping consistent with seaborn's default palette -----------------
palette = sns.color_palette()
acc_color = {
    'Correct':   palette[0],   # same blue as main line
    'Incorrect': palette[1],   # same orange as main line
}

# --- Add horizontal baselines with same colors but dotted style -------------
for r, rel in enumerate(row_order):
    for c, area in enumerate(col_order):
        ax = g.axes[r, c]
        for acc in ['Correct', 'Incorrect']:
            row = baseline_mean[
                (baseline_mean['Rel. Level Short'] == rel) &
                (baseline_mean['Accuracy Type'] == acc)
            ]
            if row.empty:
                continue
            y0 = float(row['Scene Output Similarity'])
            ax.axhline(
                y=y0,
                linestyle=':',
                linewidth=1.5,
                color=acc_color[acc],
                alpha=0.9,
                zorder=1
            )
# --- Formatting --------------------------------------------------------------
for ax in g.axes.flat:
    ax.set_xticks([0.5, 1.0])
    ax.set_xlim(0.45, 1.05)  # prevents squeezing at the left
    ax.set_ylim(0.78, 0.86)

    ax.tick_params(axis='both', labelsize=8)

    # light horizontal grid only
    ax.grid(True, axis='y')
    ax.grid(False, axis='x')

g.set_titles(row_template="Rel={row_name}", col_template="Area={col_name}")
g.set_axis_labels("Noise Level", "Semantic Similarity")

g.fig.subplots_adjust(top=0.90, hspace=0.25, wspace=0.15)
g.fig.suptitle("Scene–Output Text-based Semantic Similarity", fontsize=11)

legend = g._legend
legend.set_title("")
legend.set_frame_on(False)
legend.set_bbox_to_anchor((1.02, 0.5))
legend._loc = 10
for text in legend.texts:
    text.set_fontsize(8)

plt.show()


#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# ---- 1. Compute Correct - Incorrect difference -----------------------------
# Assumes merged_accuracy_similarity has columns "Correct" and "Incorrect"
df_delta = (
    merged_accuracy_similarity
    .assign(Delta=lambda d: d["Correct"] - d["Incorrect"])
    .groupby(["Noise Level", "Noise Area", "Rel. Level"], as_index=False)["Delta"]
    .mean()
)

# Optional: order / shorten labels for readability
rel_order = ["original", "same target", "high", "medium", "low"]
area_order = ["–", "target", "context", "all"]

df_delta["Rel. Level"] = pd.Categorical(df_delta["Rel. Level"],
                                        categories=rel_order, ordered=True)
df_delta["Noise Area"] = pd.Categorical(df_delta["Noise Area"],
                                        categories=area_order, ordered=True)

# Short labels for plotting
rel_short = {
    "original": "orig",
    "same target": "sameT",
    "high": "high",
    "medium": "med",
    "low": "low",
}
area_short = {"–": "none", "target": "tgt", "context": "ctx", "all": "all"}

df_delta["Rel_short"] = df_delta["Rel. Level"].map(rel_short)
df_delta["Area_short"] = df_delta["Noise Area"].map(area_short)

noise_levels = sorted(df_delta["Noise Level"].unique())

# ---- 2. NeurIPS-style plotting setup --------------------------------------
sns.set_theme(
    style="white",
    context="paper",
    rc={
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
    },
)

n_cols = len(noise_levels)
fig, axes = plt.subplots(
    1, n_cols,
    figsize=(3.2 * n_cols, 3.2),
    sharey=True
)

# ---- 3. Draw one heatmap per noise level ----------------------------------
vmax = df_delta["Delta"].abs().max()

for i, nl in enumerate(noise_levels):
    ax = axes[i] if n_cols > 1 else axes

    sub = df_delta[df_delta["Noise Level"] == nl]
    mat = sub.pivot(index="Rel_short", columns="Area_short", values="Delta")

    sns.heatmap(
        mat,
        ax=ax,
        vmin=-vmax,
        vmax=vmax,
        center=0,
        cmap="coolwarm",
        cbar=(i == n_cols - 1),      # single colorbar on the right
        cbar_kws={"shrink": 0.8, "label": "Δ similarity (Correct − Incorrect)"}
    )

    ax.set_title(f"Noise level = {nl}", fontsize=9)
    ax.set_xlabel("Noise area", fontsize=8)
    if i == 0:
        ax.set_ylabel("Rel. level", fontsize=8)
    else:
        ax.set_ylabel("")

    ax.tick_params(axis="both", labelsize=8)

fig.suptitle(
    "Effect of Noise on Scene–Output Semantic Separation\n"
    "(Correct − Incorrect similarity)",
    fontsize=11,
    y=1.02
)

plt.tight_layout()
plt.show()


#%%
# Group data by 'rel_level', 'Noise Level', and 'condition', then compute mean scores
semantic_by_combined = df.groupby(['Rel. Level', 'Noise Level', 'Noise Area'])[
    ['long_caption_scores', 'long_caption_text_similarity_scores']
].mean().reset_index()
semantic_by_combined

#%% Semantic Similarity Analysis
# List of scores to visualize
scores = ['long_caption_scores']

## Generate line plots for each score metric
#for score in scores:
#    plt.figure(figsize=(14, 7))
#    sns.lineplot(
#        data=semantic_by_combined,
#        x='Noise Level',
#        y=score,
#        hue='Rel. Level',
#        style='Noise Area',  # Differentiates between conditions
#        markers=True,
#        palette='tab10'
#    )
#    plt.title(f'RefCLIPScore vs Noise Level by Noise Area and Relatedness Level')
#    plt.xlabel('Noise Level')
#    plt.xticks([0,0.5,1])
#    plt.ylabel('RefCLIPScore')
#    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#    plt.tight_layout()
#    plt.show()
#
##Small-multiples with per-condition coloured baselines
#import matplotlib.pyplot as plt
#
## compute zero-noise baselines per relatedness level (in target area)
#baselines = (
#    semantic_by_combined
#    .query("`Noise Area`=='target' and `Noise Level`==0")
#    .groupby('Rel. Level')['long_caption_scores']
#    .mean()
#    .to_dict()
#)
#
#areas = semantic_by_combined['Noise Area'].unique()
#levels = semantic_by_combined['Rel. Level'].unique()
#
#fig, axes = plt.subplots(1, len(areas), figsize=(5*len(areas), 4), sharey=True)
#for ax, area in zip(axes, areas):
#    sub = semantic_by_combined[semantic_by_combined['Noise Area']==area]
#    if area == 'target':
#        sub = sub[sub['Noise Level'] > 0]  # drop zero‐noise points
#
#    for lvl in levels:
#        s = sub[sub['Rel. Level']==lvl]
#        # plot main curve and capture its color
#        line, = ax.plot(
#            s['Noise Level'], s['long_caption_scores'],
#            marker='o', label=lvl
#        )
#        color = line.get_color()
#        # draw baseline in same color
#        ax.axhline(
#            baselines[lvl],
#            linestyle='--',
#            color=color,
#            linewidth=1,
#            alpha=1,
#            zorder=0,
#            label=f"{lvl} baseline"
#        )
#    ax.set_xticks([0.5,1])
#    ax.set_title(area)
#    ax.set_xlabel('Noise Level')
#    
#
## add one dashed-line legend entry for the baseline
#axes[-1].plot([], [], linestyle='--', color='gray', label='0 noise condition')
## Set ylabel only once (shared y-axis)
#axes[0].set_ylabel('RefCLIPScore')
#
## build legend with one entry per relatedness + the baseline
#handles, labels = axes[-1].get_legend_handles_labels()
#by_label = dict(zip(labels[::2], handles[::2]))
#axes[-1].legend(
#    by_label.values(),
#    by_label.keys(),
#    bbox_to_anchor=(1.05, 1),
#    loc='upper left'
#)
#plt.suptitle("RefCLIPScore vs Noise Level by Noise Area and Relatedness Level", fontsize=16, y=1.02)
#
#plt.tight_layout()
#plt.show()
#


rng = np.random.default_rng(101)  # fixed seed for reproducibility
jitter_step = 0.08
similarity_threshold = 0.01
markersize = 6
linewidth=1.5

# Prepare radar data: include Noise Level
radar_data = (
    semantic_by_combined
    .groupby(['Noise Area', 'Noise Level', 'Rel. Level'])['long_caption_scores']
    .mean()
    .unstack('Rel. Level')  # Columns: Relatedness Level
    .reset_index()
)

categories = semantic_by_combined['Rel. Level'].unique().tolist()
num_vars = len(categories)
num_vars = len(categories)
base_angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

# --- PRECOMPUTE VALUE-BASED JITTER OFFSETS (per series x angle) ---

# matrix of shape (n_series, n_angles) with the radial values
values_matrix = radar_data[categories].to_numpy()
n_series = values_matrix.shape[0]

# initialize jitter offsets to zero
jitter_offsets = np.zeros_like(values_matrix)

for j in range(num_vars):
    col = values_matrix[:, j]           # values for angle j across all series
    order = np.argsort(col)            # sort by value
    sorted_vals = col[order]
    
    # pairwise differences
    if len(sorted_vals) > 1:
        diffs = np.diff(sorted_vals)
        min_diff = np.min(diffs)
    else:
        min_diff = np.inf
    
    # jitter only if some values are "too close"
    if min_diff < similarity_threshold:
        # assign symmetric offsets around 0: ..., -2s, -s, 0, +s, +2s, ...
        k = len(col)
        base_offsets = (np.arange(k) - (k - 1) / 2.0) * jitter_step
        # map these offsets back to original series order
        jitter_offsets[order, j] = base_offsets
    # else: leave jitter_offsets[:, j] as zeros (no jitter here)

# --- PLOT ---

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

for i, (_, row) in enumerate(radar_data.iterrows()):
    label = f"{row['Noise Area']} - Noise {row['Noise Level']}"
    values = np.array([row[cat] for cat in categories])
    
    # apply precomputed jitter for this series
    jit_angles = base_angles + jitter_offsets[i]
    
    # close the loop
    values_closed = np.concatenate([values, values[:1]])
    angles_closed = np.concatenate([jit_angles, jit_angles[:1]])
    
    if row['Noise Area'] == 'target' and row['Noise Level'] == 0.0:
        label = 'Noise 0'
    ax.plot(angles_closed, values_closed, marker='o', label=label, markersize=markersize, linewidth=linewidth)
    ax.fill(angles_closed, values_closed, alpha=0.1)

# Aesthetic settings
ax.set_ylim(0.60, 0.85)
ax.set_xticks(base_angles)
ax.set_xticklabels(categories, fontsize=11)
ax.tick_params(axis='x', pad=15)
ax.set_title('Mean RefCLIPScore per Relatedness Level, Noise Area, and Noise Level', y=1.1)

radial_ticks = [0.60, 0.65, 0.75, 0.80, 0.85]
ax.set_yticks(radial_ticks)
ax.set_yticklabels([str(t) for t in radial_ticks], fontsize=10)
ax.yaxis.grid(True)

# Legend sorted and placed outside the plot
handles, labels = ax.get_legend_handles_labels()
sorted_items = sorted(zip(labels, handles), key=lambda x: 0 if x[0] == 'Noise 0' else 1)
sorted_labels, sorted_handles = zip(*sorted_items)

fig.subplots_adjust(right=0.75)
ax.legend(
    sorted_handles, 
    sorted_labels, 
    loc='upper left',
    bbox_to_anchor=(1.05, 1.10),
)

plt.tight_layout()
plt.show()

# Accuracy Analysis
#
#plt.figure(figsize=(14, 7))
#sns.lineplot(
#    data=hard_accuracy_by_combined,
#    x='Noise Level',
#    y='hard_accuracy',
#    hue='Rel. Level',
#    style='Noise Area',  # Differentiates between conditions
#    markers=True,
#    palette='tab10'
#)
#plt.title("hard_accuracy".replace("_", " ").capitalize() + ' vs Noise Level by Noise Area and Rel Level')
#plt.xlabel('Noise Level')
#plt.xticks([0,0.5,1])
#plt.ylabel('hard_accuracy'.replace("_", " ").capitalize())
#plt.legend(title='Noise Area / Rel Level', bbox_to_anchor=(1.05, 1), loc='upper left')
#plt.tight_layout()
#plt.show()
#
#import matplotlib.pyplot as plt
#
## Compute zero-noise baselines per relatedness level (in target area)
#baselines = (
#    hard_accuracy_by_combined
#    .query("`Noise Area`=='target' and `Noise Level`==0")
#    .groupby('Rel. Level')['hard_accuracy']
#    .mean()
#    .to_dict()
#)
#
#areas = hard_accuracy_by_combined['Noise Area'].unique()
#levels = hard_accuracy_by_combined['Rel. Level'].unique()
#
#fig, axes = plt.subplots(1, len(areas), figsize=(5*len(areas), 4), sharey=True)
#for ax, area in zip(axes, areas):
#    sub = hard_accuracy_by_combined[hard_accuracy_by_combined['Noise Area'] == area]
#    if area == 'target':
#        sub = sub[sub['Noise Level'] > 0]  # drop zero-noise points
#
#    for lvl in levels:
#        s = sub[sub['Rel. Level'] == lvl]
#        # plot main curve and capture its color
#        line, = ax.plot(
#            s['Noise Level'], s['hard_accuracy'],
#            marker='o', label=lvl
#        )
#        color = line.get_color()
#        # draw baseline in same color
#        ax.axhline(
#            baselines[lvl],
#            linestyle='--',
#            color=color,
#            linewidth=1,
#            alpha=1,
#            zorder=0,
#            label=f"{lvl} baseline"
#        )
#    ax.set_xticks([0.5, 1])
#    ax.set_title(area)
#    ax.set_xlabel('Noise Level')
#
## add one dashed-line legend entry for the baseline
#axes[-1].plot([], [], linestyle='--', color='gray', label='0 noise condition')
#
## Set ylabel only once (shared y-axis)
#axes[0].set_ylabel('Hard Accuracy')
#
## Build legend with one entry per relatedness + the baseline
#handles, labels = axes[-1].get_legend_handles_labels()
#by_label = dict(zip(labels[::2], handles[::2]))
#axes[-1].legend(
#    by_label.values(),
#    by_label.keys(),
#    bbox_to_anchor=(1.05, 1),
#    loc='upper left'
#)
#plt.suptitle("Hard Accuracy vs Noise Level by Noise Area and Relatedness Level", fontsize=16, y=1.02)
#
#plt.tight_layout()
#plt.show()
#
## Prepare radar data: include Noise Level
#radar_data = (
#    hard_accuracy_by_combined
#    .groupby(['Noise Area', 'Noise Level', 'Rel. Level'])['hard_accuracy']
#    .mean()
#    .unstack('Rel. Level')  # Columns: Relatedness Level
#    .reset_index()
#)
#
#categories = hard_accuracy_by_combined['Rel. Level'].unique().tolist()
#num_vars = len(categories)
#angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
#angles += angles[:1]  # close the loop
#
## Create the figure
#fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
#
## Plot each (Noise Area, Noise Level) combination
#for _, row in radar_data.iterrows():
#    label = f"Area: {row['Noise Area']} - Noise: {row['Noise Level']}"
#    values = [row[cat] for cat in categories]
#    values += values[:1]
#    if row['Noise Area'] == 'target' and row['Noise Level'] == 0.0:
#        label = 'Noise 0'
#    ax.plot(angles, values, marker='o', label=label)
#    ax.fill(angles, values, alpha=0.1)
#
## Aesthetic settings
#ax.set_ylim(0, 0.8)
#ax.set_xticks(angles[:-1])
#ax.set_xticklabels(categories, fontsize=11)
#ax.tick_params(axis='x', pad=15)
#ax.set_title('Mean Hard Accuracy per Relatedness Level, Noise Area, and Noise Level', y=1.1)
#
## Radial scale (Hard Accuracy values)
#radial_ticks = [0, 0.2, 0.4, 0.6, 0.8]
#ax.set_yticks(radial_ticks)
#ax.set_yticklabels([str(t) for t in radial_ticks], fontsize=10)
#ax.yaxis.grid(True)
#
## Sort legend so 'Noise 0' appears first
#handles, labels = ax.get_legend_handles_labels()
#sorted_items = sorted(zip(labels, handles), key=lambda x: 0 if x[0] == 'Noise 0' else 1)
#sorted_labels, sorted_handles = zip(*sorted_items)
#ax.legend(sorted_handles, sorted_labels, loc='upper right', bbox_to_anchor=(1.5, 1.1))
#
#plt.tight_layout()
#plt.show()

#
#plt.figure(figsize=(14, 7))
#sns.lineplot(
#    data=soft_accuracy_by_combined,
#    x='Noise Level',
#    y='soft_accuracy',
#    hue='Rel. Level',
#    style='Noise Area',
#    markers=True,
#    palette='tab10'
#)
#plt.title('Accuracy vs Noise Level by Noise Area and Rel Level')
#plt.xlabel('Noise Level')
#plt.xticks([0, 0.5, 1])
#plt.ylabel('Soft Accuracy')
#plt.legend(title='Noise Area / Rel Level', bbox_to_anchor=(1.05, 1), loc='upper left')
#plt.tight_layout()
#plt.show()
#
## Compute zero-noise baselines per relatedness level (target area only)
#soft_baselines = (
#    soft_accuracy_by_combined
#    .query("`Noise Area`=='target' and `Noise Level`==0")
#    .groupby('Rel. Level')['soft_accuracy']
#    .mean()
#    .to_dict()
#)
#
#areas = soft_accuracy_by_combined['Noise Area'].unique()
#levels = soft_accuracy_by_combined['Rel. Level'].unique()
#
#fig, axes = plt.subplots(1, len(areas), figsize=(5*len(areas), 4), sharey=True)
#for ax, area in zip(axes, areas):
#    sub = soft_accuracy_by_combined[soft_accuracy_by_combined['Noise Area'] == area]
#    if area == 'target':
#        sub = sub[sub['Noise Level'] > 0]
#
#    for lvl in levels:
#        s = sub[sub['Rel. Level'] == lvl]
#        line, = ax.plot(
#            s['Noise Level'], s['soft_accuracy'],
#            marker='o', label=lvl
#        )
#        color = line.get_color()
#        ax.axhline(
#            soft_baselines[lvl],
#            linestyle='--',
#            color=color,
#            linewidth=1,
#            alpha=1,
#            zorder=0,
#            label=f"{lvl} baseline"
#        )
#    ax.set_xticks([0.5, 1])
#    ax.set_title(area)
#    ax.set_xlabel('Noise Level')
#
#axes[-1].plot([], [], linestyle='--', color='gray', label='0 noise condition')
#axes[0].set_ylabel('Accuracy')
#
#handles, labels = axes[-1].get_legend_handles_labels()
#by_label = dict(zip(labels[::2], handles[::2]))
#axes[-1].legend(
#    by_label.values(),
#    by_label.keys(),
#    bbox_to_anchor=(1.05, 1),
#    loc='upper left'
#)
#plt.suptitle("Accuracy vs Noise Level by Noise Area and Relatedness Level", fontsize=16, y=1.02)
#plt.tight_layout()
#plt.show()


# Prepare radar data: include Noise Level
radar_data_soft = (
    soft_accuracy_by_combined
    .groupby(['Noise Area', 'Noise Level', 'Rel. Level'])['soft_accuracy']
    .mean()
    .unstack('Rel. Level')  # Columns: Relatedness Level
    .reset_index()
)

categories = soft_accuracy_by_combined['Rel. Level'].unique().tolist()
num_vars = len(categories)
base_angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

# --- PRECOMPUTE VALUE-BASED JITTER OFFSETS (per series x angle) ---

values_matrix_soft = radar_data_soft[categories].to_numpy()  # (n_series, n_angles)
n_series_soft = values_matrix_soft.shape[0]

jitter_offsets_soft = np.zeros_like(values_matrix_soft)

for j in range(num_vars):
    col = values_matrix_soft[:, j]   # values for angle j across all series
    order = np.argsort(col)
    sorted_vals = col[order]

    if len(sorted_vals) > 1:
        diffs = np.diff(sorted_vals)
        min_diff = np.min(diffs)
    else:
        min_diff = np.inf

    # jitter only if some values are "too close"
    if min_diff < similarity_threshold:
        k = len(col)
        base_offsets = (np.arange(k) - (k - 1) / 2.0) * jitter_step
        jitter_offsets_soft[order, j] = base_offsets
    # else: keep zeros (no jitter for this angle)

# --- PLOT ---

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

for i, (_, row) in enumerate(radar_data_soft.iterrows()):
    label = f"{row['Noise Area']} - Noise {row['Noise Level']}"
    values = np.array([row[cat] for cat in categories])

    # apply precomputed jitter
    jit_angles = base_angles + jitter_offsets_soft[i]

    # close the loop
    values_closed = np.concatenate([values, values[:1]])
    angles_closed = np.concatenate([jit_angles, jit_angles[:1]])

    if row['Noise Area'] == 'target' and row['Noise Level'] == 0.0:
        label = 'Noise 0'
    ax.plot(angles_closed, values_closed, marker='o', label=label, markersize=markersize,linewidth=linewidth)
    ax.fill(angles_closed, values_closed, alpha=0.1)

# Aesthetic settings
ax.set_ylim(0, 1)
ax.set_xticks(base_angles)
ax.set_xticklabels(categories, fontsize=11)
ax.tick_params(axis='x', pad=15)
ax.set_title('Mean Accuracy per Relatedness Level, Noise Area, and Noise Level', y=1.1)

# Radial ticks
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_yticklabels([str(t) for t in [0, 0.2, 0.4, 0.6, 0.8, 1]], fontsize=10)
ax.yaxis.grid(True)

# Legend sorted and placed outside the plot
handles, labels = ax.get_legend_handles_labels()
sorted_items = sorted(zip(labels, handles), key=lambda x: 0 if x[0] == 'Noise 0' else 1)
sorted_labels, sorted_handles = zip(*sorted_items)

fig.subplots_adjust(right=0.75)
ax.legend(
    sorted_handles,
    sorted_labels,
    loc='upper left',
    bbox_to_anchor=(1.05, 1.10),
)

plt.tight_layout()
plt.show()


#%%

import matplotlib.pyplot as plt
import seaborn as sns

#%% VISUALIZE
# Filter dataset to only include rows with Noise Level == 0
df_zero_noise = df[df['Noise Level'] == 0]

# Categorize models by size
models_1 = [
    'Qwen/Qwen2.5-VL-7B-Instruct',
    'allenai/Molmo-7B-D-0924',
    'llava-hf/llava-onevision-qwen2-7b-ov-hf',
    'llava-hf/llava-onevision-qwen2-0.5b-si-hf',
]
models_2 = [
    'Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5',
    'microsoft/kosmos-2-patch14-224',
]

# Define display names for legend
display_name_map = {
    'Qwen/Qwen2.5-VL-7B-Instruct': 'Qwen2.5-VL-7B',
    'allenai/Molmo-7B-D-0924': 'Molmo-7B',
    'llava-hf/llava-onevision-qwen2-7b-ov-hf': 'LLaVA-OneVision-7B',
    'llava-hf/llava-onevision-qwen2-0.5b-si-hf': 'LLaVA-OneVision-0.5B',
    'Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5': 'XGen-MM-Phi3',
    'microsoft/kosmos-2-patch14-224': 'Kosmos-2',
}

# Define a custom color palette with updated keys
custom_palette = {
    'Qwen2.5-VL-7B': '#e6550d',
    'Molmo-7B': "#e91357",
    'LLaVA-OneVision-7B': '#a63603',
    'LLaVA-OneVision-0.5B': '#7f2704',
    'XGen-MM-Phi3': '#1f77b4',
    'Kosmos-2': '#6a51a3'
}

# Add 'Model Size' and display name columns
df_zero_noise['Model Size'] = df_zero_noise['model_name'].apply(
    lambda x: 'Big' if x in models_1 else 'Small'
)
df_zero_noise['Model Display Name'] = df_zero_noise['model_name'].map(display_name_map)

# Group data
performance_by_zero_noise = df_zero_noise.groupby(
    ['Rel. Level', 'Model Size', 'Model Display Name']
)[
    ['scores', 'text_similarity_scores', 'long_caption_scores', 'long_caption_text_similarity_scores']
].median().reset_index()

# Define x-axis and hue order
rel_level_order = ['original', 'same target', 'high', 'medium', 'low']
hue_order = [display_name_map[m] for m in models_1 + models_2]

# Plot
for score in ['long_caption_scores']:
    plt.figure(figsize=(16, 7))
    ax = sns.barplot(
        data=performance_by_zero_noise,
        x='Rel. Level',
        y=score,
        hue='Model Display Name',
        hue_order=hue_order,
        palette=custom_palette,
        order=rel_level_order,
        dodge=True
    )

    ax.yaxis.grid(True, linestyle='-', color='grey', alpha=0.3)
    ax.set_ylim(0.65, 0.85)
    ax.set_title(f'RefCLIPScore at Noise Level 0 by Relatedness Level')
    ax.set_xlabel('Relatedness Level')
    ax.set_ylabel('RefCLIPScore')
    ax.legend(title='Model', loc='upper right')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()

# %%
