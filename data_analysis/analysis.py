#%%
"""
Analysis of VLM performance on the COOCO dataset.

This script:
- Loads the final CSV with model outputs and scores.
- Filters to a set of desired models.
- Computes and visualizes:
  * Average relatedness scores
  * Soft/Hard accuracy and text-based similarity
  * Noise-area / noise-level effects on semantic similarity
  * RefCLIPScore and accuracy radar plots
  * Zero-noise performance across models
"""

import os
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# ------------------------ I/O & GLOBAL STYLE ---------------------------------

# Path to the full evaluation CSV
FILE_PATH = "/home/fmerlo/data/sceneregstorage/eval_output/dataset_final_final_complete.csv"

# Load data
df = pd.read_csv(FILE_PATH)

# Global plotting style
sns.set_context("talk")  # options: paper, notebook, talk, poster
plt.rcParams.update(
    {
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.title_fontsize": 14,
        "legend.fontsize": 12,
    }
)

# Inspect models present in the dataset
print("Models in the CSV:")
pprint(sorted(df["model_name"].unique()))

#%%
# ------------------------ QUICK TEXTUAL INSPECTION ---------------------------

# Print a small sample of rows to eyeball outputs
for i, (_, row) in enumerate(df.iterrows()):
    if i > 100:
        break

    print("####")
    print("Image:    ", row["image_name"])
    print("Target:   ", row["original_target"])
    print("Output:   ", row["output_clean"])
    print("Sim(T/O): ", row["original_target_output_similarity"])

#%%
# ------------------------ FILTER TO SELECTED MODELS --------------------------

desired_models = [
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "allenai/Molmo-7B-D-0924",
    "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    "Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5",
    "llava-hf/llava-onevision-qwen2-0.5b-si-hf",
    "microsoft/kosmos-2-patch14-224",
]

df = df[df["model_name"].isin(desired_models)].copy()
print(f"Filtered dataset size: {len(df)} rows")

#%%
# ------------------------ PRELIMINARY STATISTICS -----------------------------

# Average relatedness score per relatedness level
avg_scores = df.groupby("Rel. Level")["rel_score"].mean()
print("Average rel_score per Rel. Level:")
print(avg_scores)

# Prepare for plotting with ordered levels
rel_order = ["original", "same target", "high", "medium", "low"]
avg_scores_df = avg_scores.reset_index()
avg_scores_df["Rel. Level"] = pd.Categorical(
    avg_scores_df["Rel. Level"], categories=rel_order, ordered=True
)
avg_scores_df = avg_scores_df.sort_values("Rel. Level")

# Plot average rel_score by Rel. Level
plt.figure()
sns.barplot(
    data=avg_scores_df,
    x="Rel. Level",
    y="rel_score",
    palette="Blues_d",
    edgecolor="black",
)
plt.title("Average Rel. Score by Rel. Level")
plt.xlabel("Rel. Level")
plt.ylabel("Average Rel. Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%%
# ------------------------ ACCURACY METRICS (SOFT / HARD) ---------------------

# Soft Accuracy grouped by Rel. Level, Noise Level, Noise Area
soft_accuracy_by_combined = (
    df.groupby(["Rel. Level", "Noise Level", "Noise Area"])[["soft_accuracy"]]
    .mean()
    .reset_index()
)

# For Hard Accuracy we ensure we only consider outputs up to a maximum length
max_length = df["target_clean"].apply(len).max()
print("Max length of target_clean:", max_length)

# Filter rows where output length is <= max_length (matching the original heuristic)
df_hard_accuracy = df[df["output_clean"].apply(len) <= max_length].copy()

hard_accuracy_by_combined = (
    df_hard_accuracy.groupby(["Rel. Level", "Noise Level", "Noise Area"])[
        ["hard_accuracy"]
    ]
    .mean()
    .reset_index()
)

# Prepare a table-friendly summary (for LaTeX)
df_to_print = (
    df_hard_accuracy.groupby(
        ["model_name", "Rel. Level", "Noise Level", "Noise Area"]
    )[
        [
            "long_caption_scores",
            "long_caption_text_similarity_scores",
            "hard_accuracy",
            "soft_accuracy",
        ]
    ]
    .mean()
    .reset_index()
)

df_to_print.columns = [
    "model_name",
    "Rel. Level",
    "Noise Level",
    "Noise Area",
    "refCLIPScore",
    "Text-Based Similarity",
    "Hard Acc.",
    "Soft Acc.",
]

# Print one LaTeX table per model
for model, group in df_to_print.groupby("model_name"):
    print("\n")
    print(r"\begin{table}[h]")
    print(r"\hspace{-1.5cm}")
    print(group.drop(columns="model_name").to_latex(index=False, float_format="%.3f"))
    print(rf"\caption{{Results for model: {model}}}")
    print(r"\end{table}")

#%%
# ------------------------ SOFT ACCURACY & SIMILARITY -------------------------

# Mark rows where soft accuracy is exactly 1
df["correct_soft"] = df["soft_accuracy"] == 1

# Average scene-output similarity split by correctness
soft_accuracy_similarity = (
    df.groupby(["Noise Area", "Noise Level", "Rel. Level", "correct_soft"])[
        ["scene_output_similarity"]
    ]
    .mean()
    .unstack()
)

soft_accuracy_similarity.columns = ["Incorrect", "Correct"]
soft_accuracy_similarity = soft_accuracy_similarity[["Correct", "Incorrect"]]

merged_accuracy_similarity = soft_accuracy_similarity.reset_index()

# Remove impossible combinations:
#  - Noise Level == 0.0 and Noise Area != 'target'
merged_accuracy_similarity = merged_accuracy_similarity[
    ~(
        (merged_accuracy_similarity["Noise Level"] == 0.0)
        & (merged_accuracy_similarity["Noise Area"] != "target")
    )
].copy()

# Use '--' as a marker for "no noise" on the target
mask_zero_target = (merged_accuracy_similarity["Noise Level"] == 0.0) & (
    merged_accuracy_similarity["Noise Area"] == "target"
)
merged_accuracy_similarity.loc[mask_zero_target, "Noise Area"] = "--"

# Define desired ordering of categorical factors
desired_order_area = ["--", "target", "context", "all"]
desired_order_level = [0.0, 0.5, 1.0]
desired_order_rel = ["original", "same target", "high", "medium", "low"]

merged_accuracy_similarity["Noise Area"] = pd.Categorical(
    merged_accuracy_similarity["Noise Area"],
    categories=desired_order_area,
    ordered=True,
)
merged_accuracy_similarity["Noise Level"] = pd.Categorical(
    merged_accuracy_similarity["Noise Level"],
    categories=desired_order_level,
    ordered=True,
)
merged_accuracy_similarity["Rel. Level"] = pd.Categorical(
    merged_accuracy_similarity["Rel. Level"],
    categories=desired_order_rel,
    ordered=True,
)

merged_accuracy_similarity = merged_accuracy_similarity.set_index(
    ["Noise Area", "Noise Level", "Rel. Level"]
)
merged_accuracy_similarity = merged_accuracy_similarity.round(3).reset_index()

print("Merged accuracy & similarity (head):")
print(merged_accuracy_similarity.head())

#%%
# ------------------------ FACETTED BARPLOTS (SCENE SIMILARITY) ---------------

# Long format for seaborn
df_melted = merged_accuracy_similarity.melt(
    id_vars=["Noise Area", "Noise Level", "Rel. Level"],
    var_name="Accuracy Type",
    value_name="Scene Output Similarity",
)

plt.figure(figsize=(14, 8))
g = sns.catplot(
    data=df_melted,
    x="Noise Level",
    y="Scene Output Similarity",
    hue="Accuracy Type",
    col="Noise Area",
    row="Rel. Level",
    kind="bar",
    height=4,
    aspect=1.2,
    sharex=False,
)

# Customize subplots
for idx, ax in enumerate(g.axes.flat):
    ax.set_ylim(0.77, 0.86)
    # Slightly different x margings depending on noise area
    if idx in [0, 4, 8, 12, 16]:
        ax.set_xticks([0])
        ax.margins(x=0.9)
    else:
        ax.set_xticks([1, 2])
        ax.margins(x=0.1)

    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(16)
    ax.tick_params(axis="both", labelsize=14)
    ax.title.set_size(14)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

g.set_axis_labels("Noise Level", "Semantic Similarity")
plt.subplots_adjust(top=0.9)
plt.suptitle("Scene-Output Text Based Semantic Similarity", fontsize=24)

legend = g._legend
if legend:
    plt.setp(legend.get_texts(), fontsize=18)
    legend.set_title("", prop={"size": 20})
    legend.set_bbox_to_anchor((1.06, 0.5))

plt.show()

#%%
# ------------------------ RELATIONAL LINES + BASELINES -----------------------

# Re-use df_melted for a more compact line-based visualization
df_melted = merged_accuracy_similarity.melt(
    id_vars=["Noise Area", "Noise Level", "Rel. Level"],
    var_name="Accuracy Type",
    value_name="Scene Output Similarity",
)
df_melted["Noise Level"] = df_melted["Noise Level"].astype(float)

# Shorter labels for readability (Rel. level and Noise area)
df_melted["Rel. Level Short"] = df_melted["Rel. Level"].replace(
    {
        "original": "original",
        "same target": "same target",
        "high": "high",
        "medium": "medium",
        "low": "low",
    }
)

df_melted["Noise Area Short"] = df_melted["Noise Area"].replace(
    {
        "target": "target",
        "context": "context",
        "all": "all",
        # '--' is kept as-is as the baseline / no-noise label
    }
)

# Baseline rows: no noise (Noise Area '--' or equivalent)
baseline_mask = df_melted["Noise Area"].isin(["--", "–", "none"])
baseline = df_melted[baseline_mask].copy()
df_plot = df_melted[~baseline_mask].copy()

row_order = ["original", "same target", "high", "medium", "low"]
col_order = ["target", "context", "all"]

df_plot["Rel. Level Short"] = pd.Categorical(
    df_plot["Rel. Level Short"], categories=row_order, ordered=True
)
df_plot["Noise Area Short"] = pd.Categorical(
    df_plot["Noise Area Short"], categories=col_order, ordered=True
)

# One baseline value per (Rel, Accuracy Type)
baseline_mean = (
    baseline.groupby(["Rel. Level Short", "Accuracy Type"], as_index=False)[
        "Scene Output Similarity"
    ]
    .mean()
)

# Style and main relplot
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
    },
)

g = sns.relplot(
    data=df_plot,
    x="Noise Level",
    y="Scene Output Similarity",
    hue="Accuracy Type",
    style="Accuracy Type",
    kind="line",
    markers=True,
    linewidth=1.5,
    markersize=5,
    col="Noise Area Short",
    row="Rel. Level Short",
    col_order=col_order,
    row_order=row_order,
    height=2.1,
    aspect=1.4,
    facet_kws={"margin_titles": True},
)

# Map Accuracy Type -> color (consistent with seaborn defaults)
palette = sns.color_palette()
acc_color = {
    "Correct": palette[0],
    "Incorrect": palette[1],
}

# Horizontal baselines for each subplot
for r, rel in enumerate(row_order):
    for c, area in enumerate(col_order):
        ax = g.axes[r, c]
        for acc in ["Correct", "Incorrect"]:
            row = baseline_mean[
                (baseline_mean["Rel. Level Short"] == rel)
                & (baseline_mean["Accuracy Type"] == acc)
            ]
            if row.empty:
                continue
            y0 = float(row["Scene Output Similarity"])
            ax.axhline(
                y=y0,
                linestyle=":",
                linewidth=1.5,
                color=acc_color[acc],
                alpha=0.9,
                zorder=1,
            )

for ax in g.axes.flat:
    ax.set_xticks([0.5, 1.0])
    ax.set_xlim(0.45, 1.05)
    ax.set_ylim(0.78, 0.86)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")

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
# ------------------------ HEATMAP OF Δ SIMILARITY ----------------------------

# Δ = Correct − Incorrect similarity
df_delta = (
    merged_accuracy_similarity.assign(
        Delta=lambda d: d["Correct"] - d["Incorrect"]
    )
    .groupby(["Noise Level", "Noise Area", "Rel. Level"], as_index=False)["Delta"]
    .mean()
)

rel_order = ["original", "same target", "high", "medium", "low"]
area_order = ["–", "target", "context", "all"]

df_delta["Rel. Level"] = pd.Categorical(
    df_delta["Rel. Level"], categories=rel_order, ordered=True
)
df_delta["Noise Area"] = pd.Categorical(
    df_delta["Noise Area"], categories=area_order, ordered=True
)

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
fig, axes = plt.subplots(1, n_cols, figsize=(3.2 * n_cols, 3.2), sharey=True)

# Symmetric color range around 0
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
        cbar=(i == n_cols - 1),
        cbar_kws={
            "shrink": 0.8,
            "label": "Δ similarity (Correct − Incorrect)",
        },
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
    y=1.02,
)
plt.tight_layout()
plt.show()

#%%
# ------------------------ SEMANTIC SIMILARITY (RefCLIPScore) -----------------

semantic_by_combined = (
    df.groupby(["Rel. Level", "Noise Level", "Noise Area"])[
        ["long_caption_scores", "long_caption_text_similarity_scores"]
    ]
    .mean()
    .reset_index()
)

print("Semantic similarity (head):")
print(semantic_by_combined.head())

#%%
# ------------------------ RADAR PLOT: RefCLIPScore ---------------------------

rng = np.random.default_rng(101)  # fixed seed for reproducibility
jitter_step = 0.08
similarity_threshold = 0.01
markersize = 6
linewidth = 1.5

# Radar data: mean RefCLIPScore per (Noise Area, Noise Level, Rel. Level)
radar_data = (
    semantic_by_combined.groupby(["Noise Area", "Noise Level", "Rel. Level"])[
        "long_caption_scores"
    ]
    .mean()
    .unstack("Rel. Level")
    .reset_index()
)

categories = semantic_by_combined["Rel. Level"].unique().tolist()
num_vars = len(categories)
base_angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

# Pre-compute angle jitter to avoid overplotting when values are too close
values_matrix = radar_data[categories].to_numpy()
n_series = values_matrix.shape[0]
jitter_offsets = np.zeros_like(values_matrix)

for j in range(num_vars):
    col = values_matrix[:, j]
    order = np.argsort(col)
    sorted_vals = col[order]

    if len(sorted_vals) > 1:
        diffs = np.diff(sorted_vals)
        min_diff = np.min(diffs)
    else:
        min_diff = np.inf

    if min_diff < similarity_threshold:
        k = len(col)
        base_offsets = (np.arange(k) - (k - 1) / 2.0) * jitter_step
        jitter_offsets[order, j] = base_offsets

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

for i, (_, row) in enumerate(radar_data.iterrows()):
    label = f"{row['Noise Area']} - Noise {row['Noise Level']}"
    values = np.array([row[cat] for cat in categories])

    jit_angles = base_angles + jitter_offsets[i]
    values_closed = np.concatenate([values, values[:1]])
    angles_closed = np.concatenate([jit_angles, jit_angles[:1]])

    if row["Noise Area"] == "target" and row["Noise Level"] == 0.0:
        label = "Noise 0"
    ax.plot(
        angles_closed,
        values_closed,
        marker="o",
        label=label,
        markersize=markersize,
        linewidth=linewidth,
    )
    ax.fill(angles_closed, values_closed, alpha=0.1)

ax.set_ylim(0.60, 0.85)
ax.set_xticks(base_angles)
ax.set_xticklabels(categories, fontsize=11)
ax.tick_params(axis="x", pad=15)
ax.set_title(
    "Mean RefCLIPScore per Relatedness Level, Noise Area, and Noise Level", y=1.1
)

radial_ticks = [0.60, 0.65, 0.75, 0.80, 0.85]
ax.set_yticks(radial_ticks)
ax.set_yticklabels([str(t) for t in radial_ticks], fontsize=10)
ax.yaxis.grid(True)

handles, labels = ax.get_legend_handles_labels()
sorted_items = sorted(
    zip(labels, handles), key=lambda x: 0 if x[0] == "Noise 0" else 1
)
sorted_labels, sorted_handles = zip(*sorted_items)

fig.subplots_adjust(right=0.75)
ax.legend(
    sorted_handles,
    sorted_labels,
    loc="upper left",
    bbox_to_anchor=(1.05, 1.10),
)

plt.tight_layout()
plt.show()

#%%
# ------------------------ RADAR PLOT: SOFT ACCURACY --------------------------

radar_data_soft = (
    soft_accuracy_by_combined.groupby(["Noise Area", "Noise Level", "Rel. Level"])[
        "soft_accuracy"
    ]
    .mean()
    .unstack("Rel. Level")
    .reset_index()
)

categories = soft_accuracy_by_combined["Rel. Level"].unique().tolist()
num_vars = len(categories)
base_angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

values_matrix_soft = radar_data_soft[categories].to_numpy()
n_series_soft = values_matrix_soft.shape[0]
jitter_offsets_soft = np.zeros_like(values_matrix_soft)

for j in range(num_vars):
    col = values_matrix_soft[:, j]
    order = np.argsort(col)
    sorted_vals = col[order]

    if len(sorted_vals) > 1:
        diffs = np.diff(sorted_vals)
        min_diff = np.min(diffs)
    else:
        min_diff = np.inf

    if min_diff < similarity_threshold:
        k = len(col)
        base_offsets = (np.arange(k) - (k - 1) / 2.0) * jitter_step
        jitter_offsets_soft[order, j] = base_offsets

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

for i, (_, row) in enumerate(radar_data_soft.iterrows()):
    label = f"{row['Noise Area']} - Noise {row['Noise Level']}"
    values = np.array([row[cat] for cat in categories])

    jit_angles = base_angles + jitter_offsets_soft[i]
    values_closed = np.concatenate([values, values[:1]])
    angles_closed = np.concatenate([jit_angles, jit_angles[:1]])

    if row["Noise Area"] == "target" and row["Noise Level"] == 0.0:
        label = "Noise 0"
    ax.plot(
        angles_closed,
        values_closed,
        marker="o",
        label=label,
        markersize=markersize,
        linewidth=linewidth,
    )
    ax.fill(angles_closed, values_closed, alpha=0.1)

ax.set_ylim(0, 1)
ax.set_xticks(base_angles)
ax.set_xticklabels(categories, fontsize=11)
ax.tick_params(axis="x", pad=15)
ax.set_title(
    "Mean Accuracy per Relatedness Level, Noise Area, and Noise Level", y=1.1
)

ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_yticklabels([str(t) for t in [0, 0.2, 0.4, 0.6, 0.8, 1]], fontsize=10)
ax.yaxis.grid(True)

handles, labels = ax.get_legend_handles_labels()
sorted_items = sorted(
    zip(labels, handles), key=lambda x: 0 if x[0] == "Noise 0" else 1
)
sorted_labels, sorted_handles = zip(*sorted_items)

fig.subplots_adjust(right=0.75)
ax.legend(
    sorted_handles,
    sorted_labels,
    loc="upper left",
    bbox_to_anchor=(1.05, 1.10),
)

plt.tight_layout()
plt.show()

#%%
# ------------------------ ZERO-NOISE MODEL COMPARISON ------------------------

# Restrict to zero-noise rows and copy to avoid chained assignment warnings
df_zero_noise = df[df["Noise Level"] == 0].copy()

# Group models into 7B-like vs smaller
models_big = [
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "allenai/Molmo-7B-D-0924",
    "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    "llava-hf/llava-onevision-qwen2-0.5b-si-hf",
]
models_small = [
    "Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5",
    "microsoft/kosmos-2-patch14-224",
]

df_zero_noise["Model Size"] = df_zero_noise["model_name"].apply(
    lambda x: "Big" if x in models_big else "Small"
)

# Pretty display names for legend
display_name_map = {
    "Qwen/Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B",
    "allenai/Molmo-7B-D-0924": "Molmo-7B",
    "llava-hf/llava-onevision-qwen2-7b-ov-hf": "LLaVA-OneVision-7B",
    "llava-hf/llava-onevision-qwen2-0.5b-si-hf": "LLaVA-OneVision-0.5B",
    "Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5": "XGen-MM-Phi3",
    "microsoft/kosmos-2-patch14-224": "Kosmos-2",
}

df_zero_noise["Model Display Name"] = df_zero_noise["model_name"].map(
    display_name_map
)

# Color palette keyed by display name
custom_palette = {
    "Qwen2.5-VL-7B": "#e6550d",
    "Molmo-7B": "#e91357",
    "LLaVA-OneVision-7B": "#a63603",
    "LLaVA-OneVision-0.5B": "#7f2704",
    "XGen-MM-Phi3": "#1f77b4",
    "Kosmos-2": "#6a51a3",
}

performance_by_zero_noise = (
    df_zero_noise.groupby(
        ["Rel. Level", "Model Size", "Model Display Name"]
    )[
        [
            "scores",
            "text_similarity_scores",
            "long_caption_scores",
            "long_caption_text_similarity_scores",
        ]
    ]
    .median()
    .reset_index()
)

rel_level_order = ["original", "same target", "high", "medium", "low"]
hue_order = [
    display_name_map[m] for m in models_big + models_small
]

# Plot median RefCLIPScore at Noise Level 0 by relatedness and model
for score in ["long_caption_scores"]:
    plt.figure(figsize=(16, 7))
    ax = sns.barplot(
        data=performance_by_zero_noise,
        x="Rel. Level",
        y=score,
        hue="Model Display Name",
        hue_order=hue_order,
        palette=custom_palette,
        order=rel_level_order,
        dodge=True,
    )

    ax.yaxis.grid(True, linestyle="-", color="grey", alpha=0.3)
    ax.set_ylim(0.65, 0.85)
    ax.set_title("RefCLIPScore at Noise Level 0 by Relatedness Level")
    ax.set_xlabel("Relatedness Level")
    ax.set_ylabel("RefCLIPScore")
    ax.legend(title="Model", loc="upper right")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show()


#%%
# ------------------------ MIXED-EFFECTS MODELS -------------------------------
import pandas as pd
import statsmodels.formula.api as smf

# ------------------------------------------------------------------
# Aggregate data
# ------------------------------------------------------------------
agg_cols = ["image_name", "Rel. Level", "Noise Level", "Noise Area", "model_name"]
metrics = ["long_caption_scores"]

df_agg = (
    df.groupby(agg_cols, as_index=False)[metrics]
      .mean()
)

# Remove suffixes from image names (if needed)
df_agg["image_name"] = df_agg["image_name"].str.split("_").str[0]

# ------------------------------------------------------------------
# Clone zero–noise rows so that we have all noise_area conditions
# ------------------------------------------------------------------
df_zero = df_agg[df_agg["Noise Level"] == 0.0].copy()

df_agg_c = df_zero.copy()
df_agg_a = df_zero.copy()

df_agg_c["Noise Area"] = "context"
df_agg_a["Noise Area"] = "all"

df_agg = pd.concat([df_agg, df_agg_c, df_agg_a], ignore_index=True)

# ------------------------------------------------------------------
# Keep only semantic levels and noise conditions of interest
# ------------------------------------------------------------------
rel_order         = ["original", "low"]
noise_level_order = [0.0, 0.5, 1.0]  # noise level will be modeled as continuous
noise_area_order  = ["all", "context", "target"]

df_agg = df_agg[df_agg["Rel. Level"].isin(rel_order)].copy()
df_agg = df_agg[df_agg["Noise Level"].isin(noise_level_order)].copy()
df_agg = df_agg[df_agg["Noise Area"].isin(noise_area_order)].copy()

# ------------------------------------------------------------------
# 1) Tidy names
# ------------------------------------------------------------------
df_agg = df_agg.rename(columns={
    "Rel. Level":           "rel_level",
    "Noise Level":          "noise_level",
    "Noise Area":           "noise_area",
    "long_caption_scores":  "refclip"
})

# ------------------------------------------------------------------
# 2) Set types: rel_level & noise_area categorical, noise_level continuous
# ------------------------------------------------------------------
df_agg["rel_level"] = pd.Categorical(
    df_agg["rel_level"],
    categories=rel_order,
    ordered=True
)

df_agg["noise_area"] = pd.Categorical(
    df_agg["noise_area"],
    categories=noise_area_order,
    ordered=True
)

# noise_level as continuous predictor
# noise level continuous, centred at 0.5 (and scaled if you want)
df_agg["noise_level"] = df_agg["noise_level"].astype(float)

# centre at 0.5
df_agg["noise_level"] = df_agg["noise_level"] - 0.5
df_agg["noise_level"] = df_agg["noise_level"] / df_agg["noise_level"].std()

# ------------------------------------------------------------------
# 3) Balance images: keep only images with all 3×3 = 9 (rel_level, noise_area) combos
# ------------------------------------------------------------------
counts = df_agg.groupby("image_name").apply(
    lambda g: g.drop_duplicates(["rel_level", "noise_area"]).shape[0]
)

valid_images = counts[counts == len(rel_order)*len(noise_area_order)].index
df_balanced = df_agg[df_agg["image_name"].isin(valid_images)].copy()

# ------------------------------------------------------------------
# 4) Mixed-effects model
#    - Fixed: rel_level * noise_area * noise_level (noise_level continuous)
#    - Random intercepts:
#        * image_name  (groups)
#        * model_name  (variance component)
# ------------------------------------------------------------------
formula = "refclip ~ rel_level * noise_area * noise_level"

md = smf.mixedlm(
    formula,
    df_balanced,
    groups=df_balanced["image_name"],                 # random intercept for images
    vc_formula={"model": "0 + C(model_name)"}         # random intercept for models
)

m = md.fit(method="lbfgs")

print(m.summary())
print("Converged:", m.converged)


#%%
m_ols = smf.ols(
    "refclip ~ rel_level * noise_area * noise_level",
    data=df_balanced
).fit()

print(m_ols.summary())



"""
We further analyse these results using a Linear Mixed-Effects (LME) model, with refclip scores as the dependent variable. As fixed predictors, we include semantic relatedness (original vs.\ low), noise area (comparing context– and target–noise conditions to the `all’ baseline), and noise level (modelled as a continuous predictor, centred and scaled), together with all interactions. Random intercepts are included for images and for model identity.\footnote{A fuller random-effects structure with random slopes was considered, but it produced unstable estimates and convergence issues.}

Prior to modelling, we aggregated scores over images, semantic relatedness levels, noise areas, noise levels, and model identities. Zero-noise images were duplicated to ensure that all noise-area conditions were represented at the baseline level. Only images containing the full factorial combination of relatedness and noise-area conditions were retained, yielding a balanced dataset. Noise level was centred at 0.5 and standardised to aid interpretability and improve numerical stability.

The analysis focuses on the two semantic-relatedness extremes of the dataset—the original images and the low-relatedness variants—allowing us to assess how noise interacts with congruence between scene content and the target object. Table~\ref{tab:lme} summarises the LME results. The dominant patterns concern interactions between noise level and noise area: for original images, increasing noise in the context region is associated with a clear rise in refclip scores, while target-region noise has smaller effects. For images in the low-relatedness condition, increasing noise in the context region leads to a marked reduction in similarity between the model’s prediction and the true scene label, whereas increasing noise in the target region further attenuates performance.
"""
