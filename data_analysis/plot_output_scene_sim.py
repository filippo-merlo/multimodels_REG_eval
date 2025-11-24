#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint

# Specify the folder containing your CSV files
file_path = '/home/fmerlo/data/sceneregstorage/eval_output/dataset_final_final_complete.csv'
file_path_visions = '/home/fmerlo/data/sceneregstorage/attn_eval_output/results_att_deployment_VISIONS_complete_wscores.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)
df_visions = pd.read_csv(file_path_visions)

df = df[df['model_name'] == 'llava-hf/llava-onevision-qwen2-0.5b-si-hf']

# Select and standardize from df
df_scene_output = df[[
    "image_name",
    "long_caption_text_similarity_scores",
    "Rel. Level",
    "Noise Area",
    "scene_output_similarity",
    "Noise Level",
    "rel_score"
]]


# Select from df_visions and rename columns to match df_scene_output
df_visions_scene_output = df_visions[[
    "image_name",
    "long_caption_text_similarity_score",
    "Rel. Level",
    "Noise Area",
    "output_scene_text_similarity_scores",
    "Noise Level",
    "object.consistency"
]].rename(columns={
    "long_caption_text_similarity_score": "long_caption_text_similarity_scores",
    "output_scene_text_similarity_scores": "scene_output_similarity"
})
#%%
# Union of the two datasets
df_union = pd.concat([df_scene_output, df_visions_scene_output], ignore_index=True)

# If Noise Level == 0.0 → Noise Area = '--'
df_union.loc[df_union["Noise Level"] == 0.0, "Noise Area"] = "--"

# Define soft accuracy based on long_caption_text_similarity_scores
df_union["soft_accuracy"] = (df_union["long_caption_text_similarity_scores"] >= 0.9).astype(int)

df_union = df_union.apply(lambda col: col.str.lower() if col.dtype == "object" else col)

df_union

#%%
# Soft Accuracy
soft_accuracy_by_combined = df_union.groupby(['Rel. Level', 'Noise Level', 'Noise Area'])[
    ['soft_accuracy']
].mean().reset_index()


df_union['correct_soft'] = df_union['soft_accuracy'] == 1

soft_accuracy_similarity = df_union.groupby(
    ['Noise Area', 'Noise Level', 'Rel. Level', 'correct_soft']
)[['scene_output_similarity']].mean().unstack()


soft_accuracy_similarity.columns = ['Incorrect', 'Correct']
soft_accuracy_similarity = soft_accuracy_similarity[['Correct', 'Incorrect']]

merged_accuracy_similarity = soft_accuracy_similarity.reset_index()

# Define the desired order
desired_order_area = ['--', 'target', 'context', 'all']
desired_order_level = [0.0, 0.5, 1.0]
desired_order_rel = ['original','same target', 'high', 'medium', 'low', 'congruent', 'incongruent']

# Convert column to categorical with the specified order
merged_accuracy_similarity['Noise Area'] = pd.Categorical(merged_accuracy_similarity['Noise Area'], categories=desired_order_area, ordered=True)
merged_accuracy_similarity['Noise Level'] = pd.Categorical(merged_accuracy_similarity['Noise Level'], categories=desired_order_level, ordered=True)
merged_accuracy_similarity['Rel. Level'] = pd.Categorical(merged_accuracy_similarity['Rel. Level'], categories=desired_order_rel, ordered=True)


merged_accuracy_similarity = merged_accuracy_similarity.set_index(['Noise Area', 'Noise Level', 'Rel. Level'])
merged_accuracy_similarity.round(3)

# Reset index to have all categories as columns for easier plotting
merged_accuracy_similarity = merged_accuracy_similarity.reset_index()

merged_accuracy_similarity

#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Set global font sizes
sns.set_context("paper")  # options: paper, notebook, talk, poster

# --- Prepare data ------------------------------------------------------------
df_melted = merged_accuracy_similarity.melt(
    id_vars=['Noise Area', 'Noise Level', 'Rel. Level'],
    var_name='Accuracy Type',
    value_name='Scene Output Similarity'
)
df_melted['Noise Level'] = df_melted['Noise Level'].astype(float)

# --- Split baseline vs other areas ------------------------------------------
# 1) try to use Noise Area = '--' / '–' / 'none'
baseline = df_melted[df_melted['Noise Area'].isin(['--', '–', 'none'])].copy()


# 2) fallback: use Noise Level = 0 as baseline if above is empty
if baseline.empty:
    baseline = df_melted[df_melted['Noise Level'] == 0.0].copy()

df_plot = df_melted[~df_melted.index.isin(baseline.index)].copy()

row_order = ['high', 'low', 'congruent', 'incongruent']
col_order = ['target', 'context', 'all']

df_plot['Rel. Level'] = pd.Categorical(df_plot['Rel. Level'],
                                       categories=row_order, ordered=True)
df_plot['Noise Area'] = pd.Categorical(df_plot['Noise Area'],
                                       categories=col_order, ordered=True)

# one baseline value per Rel × Accuracy Type
baseline_mean = (
    baseline
    .groupby(['Rel. Level', 'Accuracy Type'], as_index=False)
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
    col='Noise Area',
    row='Rel. Level',
    col_order=col_order,
    row_order=row_order,
    height=2.1,
    aspect=1.4,
    facet_kws={'margin_titles': True}
)

# --- colour mapping: same colours as main lines ------------------------------
palette = sns.color_palette()
acc_color = {'Correct': palette[0], 'Incorrect': palette[1]}

# --- Add horizontal baselines ------------------------------------------------
for r, rel in enumerate(row_order):
    for c, area in enumerate(col_order):
        ax = g.axes[r, c]
        for acc in ['Correct', 'Incorrect']:
            row = baseline_mean[
                (baseline_mean['Rel. Level'] == rel) &
                (baseline_mean['Accuracy Type'] == acc)
            ]
            if row.empty:
                continue
            y0 = float(row['Scene Output Similarity'])
            ax.axhline(
                y=y0,
                linestyle=':',
                linewidth=2.0,
                color=acc_color[acc],
                alpha=0.9,
                zorder=5,          # above grid and area-lines
            )

# --- Formatting --------------------------------------------------------------
for ax in g.axes.flat:
    ax.set_xticks([0.5, 1.0])
    ax.set_xlim(0.45, 1.05)
    ax.set_ylim(0.78, 0.86)
    ax.tick_params(axis='both', labelsize=8)
    ax.grid(True, axis='y')
    ax.grid(False, axis='x')

g.set_titles(row_template="Rel={row_name}", col_template="Area={col_name}")
g.set_axis_labels("Noise Level", "Semantic Similarity")

g.fig.subplots_adjust(top=0.90, hspace=0.25, wspace=0.15)

legend = g._legend
legend.set_title("")
legend.set_frame_on(False)
legend.set_bbox_to_anchor((1.02, 0.5))
legend._loc = 10

plt.show()

# %%
