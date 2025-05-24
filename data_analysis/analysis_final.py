#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint

#%%
# Specify the folder containing your CSV files
file_path = '/home/fmerlo/data/sceneregstorage/eval_output/dataset_final_complete.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

set(df['model_name'])
#%%

desired_models = [
    #'Qwen/Qwen2-VL-7B-Instruct',
    'Qwen/Qwen2.5-VL-7B-Instruct',
    'allenai/Molmo-7B-D-0924',
    'llava-hf/llava-onevision-qwen2-7b-ov-hf',
    'Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5',
    # smaller models
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
# Hard Accuracy
df_hard_accuracy['correct_hard'] = df_hard_accuracy['hard_accuracy'] == 1
hard_accuracy_similarity = df_hard_accuracy.groupby(
    ['Noise Area', 'Noise Level', 'Rel. Level', 'correct_hard']
)[['scene_output_similarity']].mean().unstack()
#)[['long_caption_text_similarity_scores', 'scene_output_similarity']].mean().unstack()


# Renaming columns for clarity
#hard_accuracy_similarity.columns = ['Target/Out. Incorrect', 'Target/Out. Correct', 'Scene/Out. Incorrect', 'Scene/Out. Correct']
#hard_accuracy_similarity = hard_accuracy_similarity[['Target/Out. Correct', 'Target/Out. Incorrect', 'Scene/Out. Correct', 'Scene/Out. Incorrect']]
hard_accuracy_similarity.columns = ['Incorrect', 'Correct']
hard_accuracy_similarity = hard_accuracy_similarity[['Correct', 'Incorrect']]


merged_accuracy_similarity = hard_accuracy_similarity.reset_index()
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
# Group data by 'rel_level', 'Noise Level', and 'condition', then compute mean scores
semantic_by_combined = df.groupby(['Rel. Level', 'Noise Level', 'Noise Area'])[
    ['long_caption_scores', 'long_caption_text_similarity_scores']
].mean().reset_index()
semantic_by_combined

#%% Semantic Similarity Analysis
# List of scores to visualize
scores = ['long_caption_scores']

# Generate line plots for each score metric
for score in scores:
    plt.figure(figsize=(14, 7))
    sns.lineplot(
        data=semantic_by_combined,
        x='Noise Level',
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

#Small-multiples with per-condition coloured baselines
import matplotlib.pyplot as plt

# compute zero-noise baselines per relatedness level (in target area)
baselines = (
    semantic_by_combined
    .query("`Noise Area`=='target' and `Noise Level`==0")
    .groupby('Rel. Level')['long_caption_scores']
    .mean()
    .to_dict()
)

areas = semantic_by_combined['Noise Area'].unique()
levels = semantic_by_combined['Rel. Level'].unique()

fig, axes = plt.subplots(1, len(areas), figsize=(5*len(areas), 4), sharey=True)
for ax, area in zip(axes, areas):
    sub = semantic_by_combined[semantic_by_combined['Noise Area']==area]
    if area == 'target':
        sub = sub[sub['Noise Level'] > 0]  # drop zero‐noise points

    for lvl in levels:
        s = sub[sub['Rel. Level']==lvl]
        # plot main curve and capture its color
        line, = ax.plot(
            s['Noise Level'], s['long_caption_scores'],
            marker='o', label=lvl
        )
        color = line.get_color()
        # draw baseline in same color
        ax.axhline(
            baselines[lvl],
            linestyle='--',
            color=color,
            linewidth=1,
            alpha=1,
            zorder=0,
            label=f"{lvl} baseline"
        )
    ax.set_xticks([0.5,1])
    ax.set_title(area)
    ax.set_xlabel('Noise Level')
    

# add one dashed-line legend entry for the baseline
axes[-1].plot([], [], linestyle='--', color='gray', label='0 noise condition')
# Set ylabel only once (shared y-axis)
axes[0].set_ylabel('RefCLIPScore')

# build legend with one entry per relatedness + the baseline
handles, labels = axes[-1].get_legend_handles_labels()
by_label = dict(zip(labels[::2], handles[::2]))
axes[-1].legend(
    by_label.values(),
    by_label.keys(),
    bbox_to_anchor=(1.05, 1),
    loc='upper left'
)
plt.suptitle("RefCLIPScore vs Noise Level by Noise Area and Relatedness Level", fontsize=16, y=1.02)

plt.tight_layout()
plt.show()


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
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # close the loop

# Create the figure
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

# Plot each (Noise Area, Noise Level) combination
for _, row in radar_data.iterrows():
    label = f"{row['Noise Area']} - Noise {row['Noise Level']}"
    values = [row[cat] for cat in categories]
    values += values[:1]
    if row['Noise Area'] == 'target' and row['Noise Level'] == 0.0:
        label = 'Noise 0'
    ax.plot(angles, values, marker='o', label=label)
    ax.fill(angles, values, alpha=0.1)

# Aesthetic settings
ax.set_ylim(0.60, 0.85)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.tick_params(axis='x', pad=15)
ax.set_title('Mean RefCLIPScore per Relatedness Level, Noise Area, and Noise Level', y=1.1)

radial_ticks = [0.60, 0.65, 0.75, 0.80, 0.85]
ax.set_yticks(radial_ticks)
ax.set_yticklabels([str(t) for t in radial_ticks], fontsize=10)
ax.yaxis.grid(True)

# Sort legend so 'Noise 0' appears first
handles, labels = ax.get_legend_handles_labels()
sorted_items = sorted(zip(labels, handles), key=lambda x: 0 if x[0] == 'Noise 0' else 1)
sorted_labels, sorted_handles = zip(*sorted_items)
ax.legend(sorted_handles, sorted_labels, loc='upper right', bbox_to_anchor=(1.5, 1.1))

plt.tight_layout()
plt.show()

#%% Accuracy Analysis

plt.figure(figsize=(14, 7))
sns.lineplot(
    data=hard_accuracy_by_combined,
    x='Noise Level',
    y='hard_accuracy',
    hue='Rel. Level',
    style='Noise Area',  # Differentiates between conditions
    markers=True,
    palette='tab10'
)
plt.title("hard_accuracy".replace("_", " ").capitalize() + ' vs Noise Level by Noise Area and Rel Level')
plt.xlabel('Noise Level')
plt.xticks([0,0.5,1])
plt.ylabel('hard_accuracy'.replace("_", " ").capitalize())
plt.legend(title='Noise Area / Rel Level', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# Compute zero-noise baselines per relatedness level (in target area)
baselines = (
    hard_accuracy_by_combined
    .query("`Noise Area`=='target' and `Noise Level`==0")
    .groupby('Rel. Level')['hard_accuracy']
    .mean()
    .to_dict()
)

areas = hard_accuracy_by_combined['Noise Area'].unique()
levels = hard_accuracy_by_combined['Rel. Level'].unique()

fig, axes = plt.subplots(1, len(areas), figsize=(5*len(areas), 4), sharey=True)
for ax, area in zip(axes, areas):
    sub = hard_accuracy_by_combined[hard_accuracy_by_combined['Noise Area'] == area]
    if area == 'target':
        sub = sub[sub['Noise Level'] > 0]  # drop zero-noise points

    for lvl in levels:
        s = sub[sub['Rel. Level'] == lvl]
        # plot main curve and capture its color
        line, = ax.plot(
            s['Noise Level'], s['hard_accuracy'],
            marker='o', label=lvl
        )
        color = line.get_color()
        # draw baseline in same color
        ax.axhline(
            baselines[lvl],
            linestyle='--',
            color=color,
            linewidth=1,
            alpha=1,
            zorder=0,
            label=f"{lvl} baseline"
        )
    ax.set_xticks([0.5, 1])
    ax.set_title(area)
    ax.set_xlabel('Noise Level')

# add one dashed-line legend entry for the baseline
axes[-1].plot([], [], linestyle='--', color='gray', label='0 noise condition')

# Set ylabel only once (shared y-axis)
axes[0].set_ylabel('Hard Accuracy')

# Build legend with one entry per relatedness + the baseline
handles, labels = axes[-1].get_legend_handles_labels()
by_label = dict(zip(labels[::2], handles[::2]))
axes[-1].legend(
    by_label.values(),
    by_label.keys(),
    bbox_to_anchor=(1.05, 1),
    loc='upper left'
)
plt.suptitle("Hard Accuracy vs Noise Level by Noise Area and Relatedness Level", fontsize=16, y=1.02)

plt.tight_layout()
plt.show()

# Prepare radar data: include Noise Level
radar_data = (
    hard_accuracy_by_combined
    .groupby(['Noise Area', 'Noise Level', 'Rel. Level'])['hard_accuracy']
    .mean()
    .unstack('Rel. Level')  # Columns: Relatedness Level
    .reset_index()
)

categories = hard_accuracy_by_combined['Rel. Level'].unique().tolist()
num_vars = len(categories)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # close the loop

# Create the figure
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

# Plot each (Noise Area, Noise Level) combination
for _, row in radar_data.iterrows():
    label = f"Area: {row['Noise Area']} - Noise: {row['Noise Level']}"
    values = [row[cat] for cat in categories]
    values += values[:1]
    if row['Noise Area'] == 'target' and row['Noise Level'] == 0.0:
        label = 'Noise 0'
    ax.plot(angles, values, marker='o', label=label)
    ax.fill(angles, values, alpha=0.1)

# Aesthetic settings
ax.set_ylim(0, 0.8)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.tick_params(axis='x', pad=15)
ax.set_title('Mean Hard Accuracy per Relatedness Level, Noise Area, and Noise Level', y=1.1)

# Radial scale (Hard Accuracy values)
radial_ticks = [0, 0.2, 0.4, 0.6, 0.8]
ax.set_yticks(radial_ticks)
ax.set_yticklabels([str(t) for t in radial_ticks], fontsize=10)
ax.yaxis.grid(True)

# Sort legend so 'Noise 0' appears first
handles, labels = ax.get_legend_handles_labels()
sorted_items = sorted(zip(labels, handles), key=lambda x: 0 if x[0] == 'Noise 0' else 1)
sorted_labels, sorted_handles = zip(*sorted_items)
ax.legend(sorted_handles, sorted_labels, loc='upper right', bbox_to_anchor=(1.5, 1.1))

plt.tight_layout()
plt.show()


#%%

plt.figure(figsize=(14, 7))
sns.lineplot(
    data=soft_accuracy_by_combined,
    x='Noise Level',
    y='soft_accuracy',
    hue='Rel. Level',
    style='Noise Area',
    markers=True,
    palette='tab10'
)
plt.title('Soft Accuracy vs Noise Level by Noise Area and Rel Level')
plt.xlabel('Noise Level')
plt.xticks([0, 0.5, 1])
plt.ylabel('Soft Accuracy')
plt.legend(title='Noise Area / Rel Level', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Compute zero-noise baselines per relatedness level (target area only)
soft_baselines = (
    soft_accuracy_by_combined
    .query("`Noise Area`=='target' and `Noise Level`==0")
    .groupby('Rel. Level')['soft_accuracy']
    .mean()
    .to_dict()
)

areas = soft_accuracy_by_combined['Noise Area'].unique()
levels = soft_accuracy_by_combined['Rel. Level'].unique()

fig, axes = plt.subplots(1, len(areas), figsize=(5*len(areas), 4), sharey=True)
for ax, area in zip(axes, areas):
    sub = soft_accuracy_by_combined[soft_accuracy_by_combined['Noise Area'] == area]
    if area == 'target':
        sub = sub[sub['Noise Level'] > 0]

    for lvl in levels:
        s = sub[sub['Rel. Level'] == lvl]
        line, = ax.plot(
            s['Noise Level'], s['soft_accuracy'],
            marker='o', label=lvl
        )
        color = line.get_color()
        ax.axhline(
            soft_baselines[lvl],
            linestyle='--',
            color=color,
            linewidth=1,
            alpha=1,
            zorder=0,
            label=f"{lvl} baseline"
        )
    ax.set_xticks([0.5, 1])
    ax.set_title(area)
    ax.set_xlabel('Noise Level')

axes[-1].plot([], [], linestyle='--', color='gray', label='0 noise condition')
axes[0].set_ylabel('Soft Accuracy')

handles, labels = axes[-1].get_legend_handles_labels()
by_label = dict(zip(labels[::2], handles[::2]))
axes[-1].legend(
    by_label.values(),
    by_label.keys(),
    bbox_to_anchor=(1.05, 1),
    loc='upper left'
)
plt.suptitle("Soft Accuracy vs Noise Level by Noise Area and Relatedness Level", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


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
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

# Plot each (Noise Area, Noise Level) combination
for _, row in radar_data_soft.iterrows():
    label = f"{row['Noise Area']} - Noise {row['Noise Level']}"
    values = [row[cat] for cat in categories]
    values += values[:1]
    if row['Noise Area'] == 'target' and row['Noise Level'] == 0.0:
        label = 'Noise 0'
    ax.plot(angles, values, marker='o', label=label)
    ax.fill(angles, values, alpha=0.1)

# Aesthetic settings
ax.set_ylim(0, 1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.tick_params(axis='x', pad=15)
ax.set_title('Mean Soft Accuracy per Relatedness Level, Noise Area, and Noise Level', y=1.1)

# Radial ticks
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_yticklabels([str(t) for t in [0, 0.2, 0.4, 0.6, 0.8, 1]], fontsize=10)
ax.yaxis.grid(True)

# Sort legend so 'Noise 0' appears first
handles, labels = ax.get_legend_handles_labels()
sorted_items = sorted(zip(labels, handles), key=lambda x: 0 if x[0] == 'Noise 0' else 1)
sorted_labels, sorted_handles = zip(*sorted_items)
ax.legend(sorted_handles, sorted_labels, loc='upper right', bbox_to_anchor=(1.5, 1.1))

plt.tight_layout()
plt.show()


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
    'Qwen/Qwen2-VL-7B-Instruct',
]

# Define a custom color palette
custom_palette = {
    'Qwen/Qwen2.5-VL-7B-Instruct': '#e6550d',   # warm orange
    'Qwen/Qwen2-VL-7B-Instruct': '#31a354',     # green (as requested)
    'allenai/Molmo-7B-D-0924': "#e91357",       # red
    'llava-hf/llava-onevision-qwen2-7b-ov-hf': '#a63603',  # dark orange/brown
    'llava-hf/llava-onevision-qwen2-0.5b-si-hf': '#7f2704', # deeper brown
    'Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5': '#1f77b4',  # blue (cold)
    'microsoft/kosmos-2-patch14-224': '#6a51a3'  # deep purple (cold)
}


# Add a 'Model Size' column
df_zero_noise['Model Size'] = df_zero_noise['model_name'].apply(
    lambda x: 'Big' if x in models_1 else 'Small'
)

# Add a formatted label for display (optional)
df_zero_noise['Model Display'] = df_zero_noise['Model Size'] + ' | ' + df_zero_noise['model_name']

# Group data by 'Rel. Level', 'Model Size', and 'model_name'
performance_by_zero_noise = df_zero_noise.groupby(['Rel. Level', 'Model Size', 'model_name'])[
    ['scores', 'text_similarity_scores', 'long_caption_scores', 'long_caption_text_similarity_scores']
].median().reset_index()

# Define custom x-axis order
rel_level_order = ['original', 'same target', 'high', 'medium', 'low']

# Plot
for score in ['long_caption_scores']:
    plt.figure(figsize=(16, 7))
    ax = sns.barplot(
        data=performance_by_zero_noise,
        x='Rel. Level',
        y=score,
        hue='model_name',
        hue_order=models_1 + models_2,
        palette=custom_palette,
        order=rel_level_order,
        dodge=True
    )


    # Horizontal grey lines at each y-tick
    ax.yaxis.grid(True, linestyle='-', color='grey', alpha=0.3)

    ax.set_ylim(0.5, 0.85)
    ax.set_title(f'RefCLIPScore at Noise Level 0 by Relatedness Level')
    ax.set_xlabel('Relatedness Level')
    ax.set_ylabel('RefCLIPScore')
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()
