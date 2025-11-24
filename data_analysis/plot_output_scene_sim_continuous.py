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
]].rename(columns={
    "long_caption_text_similarity_scores": "target_output_similarity"
})

df_scene_output['dataset'] = 'COOCO'

# filter out Rel. Level in ['original', 'same target']
df_scene_output = df_scene_output[~df_scene_output['Rel. Level'].isin(['original', 'same target'])]

# normalize rel_score to [0,1]
min_rel_score = df_scene_output['rel_score'].min()
max_rel_score = df_scene_output['rel_score'].max()
df_scene_output['normalized score'] = (df_scene_output['rel_score'] - min_rel_score) / (max_rel_score - min_rel_score)

df_scene_output

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
    "long_caption_text_similarity_score": "target_output_similarity",
    "output_scene_text_similarity_scores": "scene_output_similarity"
})

df_visions_scene_output['dataset'] = 'VISIONS'
# ---- PARSE object.consistency (“mean ± sd”) → extract mean ----
df_visions_scene_output["object.consistency"] = (
    df_visions_scene_output["object.consistency"]
        .str.extract(r'([0-9]*\.?[0-9]+)\s*±')[0]
        .astype(float)
)

# ---- MIN–MAX NORMALIZATION TO [0, 1] ----
col = "normalized score"
min_val = df_visions_scene_output["object.consistency"].min()
max_val = df_visions_scene_output["object.consistency"].max()

df_visions_scene_output[col] = (
    df_visions_scene_output["object.consistency"] - min_val
) / (max_val - min_val)

df_visions_scene_output

# Union of the two datasets
df_union = pd.concat([df_scene_output, df_visions_scene_output], ignore_index=True)

# If Noise Level == 0.0 → Noise Area = '--'
df_union.loc[df_union["Noise Level"] == 0.0, "Noise Area"] = "--"

# Define soft accuracy based on long_caption_text_similarity_scores
df_union["soft_accuracy"] = (df_union["target_output_similarity"] >= 0.9).astype(int)

df_union = df_union.apply(lambda col: col.str.lower() if col.dtype == "object" else col)

df_union.drop(columns=['rel_score', 'object.consistency','Rel. Level'], inplace=True)


col_order = ['--', 'target', 'context', 'all']

df_union['Noise Area'] = pd.Categorical(df_union['Noise Area'],
                                       categories=col_order, ordered=True)

df_union

#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- Bin normalized score ---
n_bins = 10
df_union["norm_bin"] = pd.cut(df_union["normalized score"], bins=n_bins)

# --- Compute means ---
binned = (
    df_union
    .groupby(["dataset", "Noise Area", "Noise Level", "soft_accuracy", "norm_bin"])
    .agg({
        "normalized score": "mean",
        "scene_output_similarity": "mean"
    })
    .reset_index()
)

binned["Noise Level"] = binned["Noise Level"].astype(str)

# ------------------------------------------------------------------
# 1) Extract the baseline curve from Noise Area == "--"
# ------------------------------------------------------------------

baseline_ref = binned[
    (binned["Noise Area"] == "--") &
    (binned["Noise Level"] == "0.0")
].copy()

def interpolate_baseline(df):
    # sort
    df = df.sort_values("normalized score").copy()
    
    # interpolate only numeric columns
    num_cols = ["normalized score", "scene_output_similarity"]
    df[num_cols] = df[num_cols].infer_objects(copy=False).interpolate(limit_direction="both")
    
    return df

baseline_ref = (
    baseline_ref
    .groupby(["dataset", "soft_accuracy"], group_keys=False)
    .apply(interpolate_baseline)
)
# ------------------------------------------------------------------
# 2) Remove Noise Area == "--" from the actual plot
# ------------------------------------------------------------------

binned_no_missing = binned[binned["Noise Area"] != "--"].copy()

# ------------------------------------------------------------------
# 3) Prepare band (0.5 / 1.0 noise)
# ------------------------------------------------------------------

high_noise = binned_no_missing[binned_no_missing["Noise Level"].isin(["0.5", "1.0"])]

band = (
    high_noise
    .groupby(["dataset", "Noise Area", "soft_accuracy", "norm_bin"])
    .agg({
        "normalized score": "mean",
        "scene_output_similarity": ["min", "max"]
    })
    .reset_index()
)

band.columns = [
    "dataset", "Noise Area", "soft_accuracy", "norm_bin",
    "normalized score", "y_min", "y_max"
]

# ------------------------------------------------------------------
# 4) Plot with baseline injected into every facet
# ------------------------------------------------------------------

sns.set(style="whitegrid")

g = sns.FacetGrid(
    binned_no_missing[binned_no_missing["Noise Level"] == "0.0"],
    row="Noise Area",
    col="dataset",
    hue="soft_accuracy",
    hue_order=[1, 0],
    margin_titles=True,
    height=3.2,
    aspect=1.2
)


def add_band(data, color, **kwargs):
    m = (
        (band["dataset"] == data["dataset"].iloc[0]) &
        (band["Noise Area"] == data["Noise Area"].iloc[0]) &
        (band["soft_accuracy"] == data["soft_accuracy"].iloc[0])
    )
    band_sub = band[m].sort_values("normalized score")
    if len(band_sub) == 0:
        return
    plt.fill_between(
        band_sub["normalized score"],
        band_sub["y_min"],
        band_sub["y_max"],
        alpha=0.20,
        color=color
    )


def add_baseline(data, color, **kwargs):
    """Plot the reference baseline from Noise Area == '--' in all facets."""
    d = baseline_ref[
        (baseline_ref["dataset"] == data["dataset"].iloc[0]) &
        (baseline_ref["soft_accuracy"] == data["soft_accuracy"].iloc[0])
    ].sort_values("normalized score")

    plt.plot(
        d["normalized score"],
        d["scene_output_similarity"],
        color=color,
        linestyle="-",
        linewidth=1.5,
        alpha=0.8
    )


# Draw high-noise band first
g.map_dataframe(add_band)

# Draw local baseline (0.0) for each Noise Area
g.map_dataframe(
    sns.lineplot,
    x="normalized score",
    y="scene_output_similarity"
)

# Add global reference baseline
g.map_dataframe(add_baseline)

g.set_axis_labels("Normalized score (binned)", "Mean scene–output similarity")
g.add_legend(title="", labels=["Correct", "Incorrect"])

legend = g._legend
legend.set_title("")
legend.set_frame_on(False)
legend.set_bbox_to_anchor((1.02, 0.5))
legend._loc = 10

plt.tight_layout()
plt.show()
