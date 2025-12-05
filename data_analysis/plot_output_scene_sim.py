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
    "rel_score",
    "scene"
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

# Union of the two datasets
df_union = pd.concat([df_scene_output, df_visions_scene_output], ignore_index=True)

# If Noise Level == 0.0 → Noise Area = '--'
df_union.loc[df_union["Noise Level"] == 0.0, "Noise Area"] = "--"

# Define soft accuracy based on long_caption_text_similarity_scores
df_union["soft_accuracy"] = (df_union["long_caption_text_similarity_scores"] >= 0.9).astype(int)

df_union = df_union.apply(lambda col: col.str.lower() if col.dtype == "object" else col)


print("Rel. Level:", df_union["Rel. Level"].unique())
print("Noise Area:", df_union["Noise Area"].unique())
print("Noise Level:", df_union["Noise Level"].unique())

print("scene_output_similarity range:", 
      df_union["scene_output_similarity"].min(), 
      df_union["scene_output_similarity"].max())

print("caption similarity range:", 
      df_union["long_caption_text_similarity_scores"].min(), 
      df_union["long_caption_text_similarity_scores"].max())

print("soft_accuracy:", df_union["soft_accuracy"].unique())

"""
Output:

Rel. Level: ['original' 'low' 'medium' 'same target' 'high' 'congruent' 'incongruent']
Noise Area: ['--' 'target' 'context' 'all']
Noise Level: [0.  0.5 1. ]
scene_output_similarity range: 0.567659318447113 1.0
caption similarity range: 0.5801180601119995 1.000000238418579
soft_accuracy: [0 1]
"""

# Soft Accuracy
soft_accuracy_by_combined = df_union.groupby(['Rel. Level', 'Noise Level', 'Noise Area'])[
    ['soft_accuracy']
].mean().reset_index()


df_union['correct_soft'] = df_union['soft_accuracy'] == 1

soft_accuracy_similarity = df_union.groupby(
    ['Noise Area', 'Noise Level', 'Rel. Level', 'correct_soft']
)[['scene_output_similarity','long_caption_text_similarity_scores']].mean().unstack()

soft_accuracy_similarity.columns = [
    'SOS - Incorrect',
    'SOS - Correct',
    'TOS - Incorrect',
    'TOS - Correct']

merged_accuracy_similarity = soft_accuracy_similarity.reset_index()

#%%
# Assume df_union has columns: 'image', 'scene'
mask = df_union['scene'].isna()

df_union.loc[mask, 'scene'] = (
    df_union.loc[mask, 'image_name']
    .astype(str)
    .str.split('_')
    .str[0]
)
df_union.to_csv("cooco_visions_scene_output_sim.csv", index=False)

#%%
print(merged_accuracy_similarity.to_csv())
#%%
df = merged_accuracy_similarity

#dep_vars = ["SOS - Incorrect", "SOS - Correct", "TOS - Incorrect", "TOS - Correct"]
dep_vars = ["SOS - Incorrect", "SOS - Correct"]

# ---------------------------------------------------------
# define conditions of interest
# ---------------------------------------------------------
rel_levels_of_interest = ["high", "low", "congruent", "incongruent"]
noise_areas_of_interest = ["target", "context", "all"]
noise_levels_of_interest = [0.5, 1.0]

# ---------------------------------------------------------
# 1) Baseline at 0 noise PER REL LEVEL
# ---------------------------------------------------------
baseline_df = df[
    (df["Noise Level"] == 0.0) &
    (df["Rel. Level"].isin(rel_levels_of_interest))
]

# baseline: index = Rel. Level, columns = dep_vars
baseline = baseline_df.groupby("Rel. Level")[dep_vars].mean()

print("Baseline at 0 noise (per Rel. Level):")
print(baseline)

# ---------------------------------------------------------
# 2) Filter data for plotting (0.5 and 1.0)
# ---------------------------------------------------------
plot_df = df[
    df["Rel. Level"].isin(rel_levels_of_interest)
    & df["Noise Area"].isin(noise_areas_of_interest)
    & df["Noise Level"].isin(noise_levels_of_interest)
].copy()

# Ensure RelLevel order
rel_order = ["high", "low", "congruent", "incongruent"]
plot_df["Rel. Level"] = pd.Categorical(plot_df["Rel. Level"],
                                       categories=rel_order,
                                       ordered=True)

# ---------------------------------------------------------
# Plot: one figure per dependent variable, three subplots (target/context/all)
# ---------------------------------------------------------
for dv in dep_vars:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    fig.suptitle(dv.split(" - ")[1])

    baseline_label_used = False  # to avoid duplicate legend entries

    for ax, area in zip(axes, noise_areas_of_interest):
        sub = plot_df[plot_df["Noise Area"] == area].sort_values(["Rel. Level", "Noise Level"])

        x_positions = range(len(rel_order))
        width = 0.25  # bar width

        # Bars for each noise level (0.5, 1.0)
        for i, nl in enumerate(noise_levels_of_interest):
            sub_nl = sub[sub["Noise Level"] == nl]

            heights = [
                sub_nl[sub_nl["Rel. Level"] == r][dv].values[0]
                for r in rel_order
            ]
            offset = (i - 0.5) * width
            colors = {
                0.5: "#777777", 
                1.0: "#000000"  
            }

            ax.bar(
                [x + offset for x in x_positions],
                heights,
                width=width,
                color=colors[nl],
                label=f"Noise {nl}"
            )

        # Baseline segments per Rel. Level
        for idx, r in enumerate(rel_order):
            if r in baseline.index:
                y = baseline.loc[r, dv]
                ax.hlines(
                    y,
                    idx - 0.45,
                    idx + 0.45,
                    linestyles="--",
                    linewidth=1,
                    color="red",
                    label="Baseline 0.0" if not baseline_label_used else None
                )
                baseline_label_used = True

        ax.set_title(f"Noise area: {area}")
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels(rel_order, rotation=30)
        if area == "target":
            ax.set_ylabel('Semantic Similarity')
        else:
            ax.set_ylabel('')
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_ylim(0.78, 0.86)

    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()


# %%
# assumes:
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

df = merged_accuracy_similarity

# dependent variables
dep_vars = ["SOS - Incorrect", "SOS - Correct"]
dv_incorrect, dv_correct = dep_vars

# ---------------------------------------------------------
# define conditions of interest
# ---------------------------------------------------------
#rel_levels_of_interest = ["original","same target","high","medium", "low", "congruent", "incongruent"]
rel_levels_of_interest = ["high","low", "congruent", "incongruent"]

noise_areas_of_interest = ["target", "context", "all"]
noise_levels_of_interest = [0.5, 1.0]

# ---------------------------------------------------------
# 1) Baseline at 0 noise PER REL LEVEL
# ---------------------------------------------------------
baseline_df = df[
    (df["Noise Level"] == 0.0) &
    (df["Rel. Level"].isin(rel_levels_of_interest))
]

# baseline: index = Rel. Level, columns = dep_vars
baseline = baseline_df.groupby("Rel. Level")[dep_vars].mean()

print("Baseline at 0 noise (per Rel. Level):")
print(baseline)

# ---------------------------------------------------------
# 2) Filter data for plotting (0.5 and 1.0)
# ---------------------------------------------------------
plot_df = df[
    df["Rel. Level"].isin(rel_levels_of_interest)
    & df["Noise Area"].isin(noise_areas_of_interest)
    & df["Noise Level"].isin(noise_levels_of_interest)
].copy()

# Ensure RelLevel order
rel_order = rel_levels_of_interest
plot_df["Rel. Level"] = pd.Categorical(
    plot_df["Rel. Level"],
    categories=rel_order,
    ordered=True
)

# ---------------------------------------------------------
# Plot: ONE figure (SOS) with three subplots, Correct & Incorrect side by side
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
fig.suptitle("")

x_positions = np.arange(len(rel_order))

# Order for each Rel. Level on x:
#   Correct 0.5, Correct 1.0, Incorrect 0.5, Incorrect 1.0
order = [
    ("Correct",   0.5),
    ("Correct",   1.0),
    ("Incorrect", 0.5),
    ("Incorrect", 1.0),
]

group_width = 0.8
n_bars = len(order)
bar_width = group_width / n_bars

# center the 4 bars around each x-position
offsets = {
    order[i]: (i - (n_bars - 1) / 2) * bar_width
    for i in range(n_bars)
}

# four distinct colors for the four bar types
colors_cols = {
    ("Correct",   0.5): "#8BB9E3",  # light blue
    ("Correct",   1.0): "#2F6FA1",  # dark blue

    ("Incorrect", 0.5): "#E9A3A3",  # light red
    ("Incorrect", 1.0): "#C23B38",  # dark red
}
# to avoid repeated legend entries for baselines
baseline_label_used = {
    "Incorrect": False,
    "Correct": False,
}

for ax, area in zip(axes, noise_areas_of_interest):
    sub = plot_df[plot_df["Noise Area"] == area].sort_values(
        ["Rel. Level", "Noise Level"]
    )

    # -------- bars --------
    for correctness, nl in order:
        dv = f"SOS - {correctness}"
        sub_nl = sub[sub["Noise Level"] == nl]

        heights = [
            sub_nl[sub_nl["Rel. Level"] == r][dv].values[0]
            for r in rel_order
        ]

        pos = [x + offsets[(correctness, nl)] for x in x_positions]

        ax.bar(
            pos,
            heights,
            width=bar_width,
            color=colors_cols[(correctness, nl)],
            label=f"{correctness}, Noise {nl}",
        )

    # -------- baselines: separate for Incorrect and Correct --------
    for idx, r in enumerate(rel_order):
        if r in baseline.index:
            # correct baseline (dashed red)
            y_cor = baseline.loc[r, dv_correct]
            ax.hlines(
                y_cor,
                idx - group_width / 2,
                idx,
                color="black",
                linestyle="-",
                linewidth=1,
                label="Correct, Baseline 0.0"
                if not baseline_label_used["Correct"] else None,
            )
            baseline_label_used["Correct"] = True

            # incorrect baseline (solid red)
            y_inc = baseline.loc[r, dv_incorrect]
            ax.hlines(
                y_inc,
                idx,
                idx + group_width / 2,
                color="black",
                linestyle="--",
                linewidth=1,
                label="Incorrect, Baseline 0.0"
                if not baseline_label_used["Incorrect"] else None,
            )
            baseline_label_used["Incorrect"] = True

            

    ax.set_title(f"Noise area: {area}")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(rel_order, rotation=30)
    # --- extra labels for dataset groups ---
    # positions between high/low and congruent/incongruent
    coco_x = 0.5 * (x_positions[0] + x_positions[-3])
    visions_x = 0.5 * (x_positions[-2] + x_positions[-1])

    # y < 0 puts text below the tick labels (axis coordinates)
    ax.text(
        coco_x,
        -0.30,
        "COOCO",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=9,
    )
    ax.text(
        visions_x,
        -0.30,
        "VISIONS",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=9,
    )

    if area == "target":
        ax.set_ylabel("Semantic Similarity")
    else:
        ax.set_ylabel("")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_ylim(0.78, 0.86)

# -------- single, deduplicated legend --------
handles, labels = axes[0].get_legend_handles_labels()
seen = set()
uniq_handles, uniq_labels = [], []
for h, l in zip(handles, labels):
    if l not in seen and l != "":
        uniq_handles.append(h)
        uniq_labels.append(l)
        seen.add(l)

fig.legend(uniq_handles, uniq_labels, loc="upper right").set_bbox_to_anchor((1.05, 0.7))
plt.tight_layout(rect=[0, -0.2, 0.9, 0.95])
plt.show()

# %%
