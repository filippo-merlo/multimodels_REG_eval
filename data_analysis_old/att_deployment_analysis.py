#%%
##################################################
# === IMPORT LIBRARIES AND INITIAL SETTINGS ===
##################################################
import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import os
from pprint import pprint

# Enable progress bars for pandas operations
tqdm.pandas()

# --- Matplotlib Style Configuration ---
plt.rcParams.update({
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.title_fontsize': 14,
    'legend.fontsize': 12,
    'font.size': 14  # base font size
})

# --- Separator for console readability ---
separator = "\n\n##################################################\n##################################################\n\n"

##################################################
# === LOAD AND PREPROCESS DATA ===
##################################################

# --- Specify CSV file path ---
file_path = '/home/fmerlo/data/sceneregstorage/attn_eval_output/results_att_deployment_last_wscores.csv'

# --- Load CSV into pandas DataFrame ---
df = pd.read_csv(file_path)
#%%
#df['rel_score'][df['rel_level'] == 'original']
#df['rel_score'][df['rel_level'] == 'same_target']
#
#for _, row in df.iterrows():
#    name = row['image_name']
#    if 'original' in name:
#        rel_name = name.replace('original', 'relscore_same_target')
#        orig_score = row['rel_score']
#        if rel_name in df['image_name'].values:
#            rel_score = df.loc[df['image_name'] == rel_name, 'rel_score'].values[0]
#            if pd.isna(rel_score):
#                # Fill missing value
#                df.loc[df['image_name'] == rel_name, 'rel_score'] = orig_score
#            elif rel_score != orig_score:
#                print(f"⚠️ Mismatch for {rel_name}: original={orig_score}, current={rel_score}")
##%%
## 1. Build mapping from original -> relscore_same_target
#orig_to_rel = {
#    row['image_name'].replace('original', 'relscore_same_target'): row['rel_score']
#    for _, row in df.iterrows()
#    if 'original' in row['image_name']
#}
#
## 2. Update relscore_same_target rows with corresponding values
#df.loc[
#    df['image_name'].isin(orig_to_rel.keys()), 'rel_score'
#] = df['image_name'].map(orig_to_rel)
#

#%%

# --- Basic cleaning and normalization ---
df = df[df['target'] != "nothing"]  # Remove rows with meaningless targets
df['Rel. Level'] = df['rel_level'].fillna('original').apply(lambda x: x.replace('_', ' '))
df = df.drop(columns=['rel_level'])
df['Rel. Level'] = df['Rel. Level'].apply(lambda x: x.replace('middle', 'medium'))

# --- Split and clean noise columns ---
df['Noise Area'] = df['condition'].apply(lambda x: x.split('_')[0])
df = df.drop(columns=['condition'])
df['Noise Level'] = df['noise_level']
df = df.drop(columns=['noise_level'])

#%%
##################################################
# === OPTIONAL: FILTER BASED ON IMAGE FILENAMES ===
##################################################
# (Commented out - can be activated if needed)
#
# filtered_images_folder_path = '/Users/filippomerlo/Desktop/manually_filtered_images'
#
# # Collect all image filenames in folder
# image_filenames = {f for f in os.listdir(filtered_images_folder_path)
#                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))}
#
# # Extract IDs from filenames
# image_filenames_id = {f.split('_')[0] for f in image_filenames}
#
# # Filter df based on image ID presence
# df = df[df['image_name'].apply(lambda x: x.split('_')[0] in image_filenames_id)] if 'image_name' in df.columns else df


#%%
#%%
##################################################
# === PARSE LIST-LIKE COLUMNS AND COMPUTE RATIOS ===
##################################################

def parse_list(value):
    """Convert string representations of lists into actual lists."""
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return []
    return value


def compute_ratio(row):
    """Compute ratio of attention over target to context per layer."""
    attn_over_target = np.array(row['attn_over_target'], dtype=np.float32)
    attn_over_context = np.array(row['attn_over_context'], dtype=np.float32)
    ratio = np.divide(attn_over_target, attn_over_context, out=np.full_like(attn_over_target, np.nan), where=attn_over_context!=0)
    return ratio.tolist()


# --- Apply list parsing and ratio computation ---
df['attn_over_target'] = df['attn_over_target'].apply(parse_list)
df['attn_over_context'] = df['attn_over_context'].apply(parse_list)
df['attn_ratio'] = df.progress_apply(compute_ratio, axis=1)

#%%
# --- Soft accuracy (based on cosine similarity threshold) ---
df['soft_accuracy'] = (df['long_caption_text_similarity_scores'] >= 0.90).astype(int)

#%%

# --- Subset: retain only no-noise condition ---
#df_no_noise = df[df['Noise Level'] == 0.0].copy()
df_no_noise = df[(df['Noise Level'] == 1.) & (df['Noise Area']=='all')].copy()

# --- Keep only correctly predicted samples ---
df_no_noise = df_no_noise[df_no_noise['soft_accuracy'] == 1].copy()

# --- Exclude 'original' and 'same target' relevance levels ---
df_no_noise = df_no_noise[
    (df_no_noise['Rel. Level'] != 'original') &
    (df_no_noise['Rel. Level'] != 'same target')
].copy()


# --- Expand lists to have one row per layer ---
df_no_noise_expanded = df_no_noise.explode('attn_ratio')
df_no_noise_expanded['layer'] = df_no_noise_expanded.groupby(level=0).cumcount()

# --- Convert attention values to float ---
df_no_noise_expanded['attn_ratio'] = pd.to_numeric(df_no_noise_expanded['attn_ratio'], errors='coerce')

# --- Compute layer-wise statistics ---
layer_stats = (
    df_no_noise_expanded
    .groupby('layer')
    .agg(
        mean_attn=('attn_ratio', 'mean'),
        std_attn=('attn_ratio', 'std')
    )
    .reset_index()
)

# --- Rank layers by mean attention and stability ---
layer_stats['cv_attn'] = layer_stats['std_attn'] / layer_stats['mean_attn']  # coefficient of variation
layer_stats_sorted = layer_stats.sort_values(by='mean_attn', ascending=False)

# --- Print summary ---
print("\n" + "="*60)
print("LAYER-WISE ATTENTION (NO NOISE)")
print("="*60)
print(layer_stats_sorted.round(3))
print("="*60 + "\n")

# --- Plot: mean and variability across layers ---
fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.plot(layer_stats['layer'], layer_stats['mean_attn'], marker='o', label='Mean attention')
ax1.fill_between(
    layer_stats['layer'],
    layer_stats['mean_attn'] - layer_stats['std_attn'],
    layer_stats['mean_attn'] + layer_stats['std_attn'],
    alpha=0.2, label='±1 std'
)
ax1.set_xlabel('Layer')
ax1.set_ylabel('Mean Attention Ratio')
ax1.set_title('Layer-wise Mean Attention Ratio (Noise = 0)')
ax1.grid(True)
ax1.legend()

plt.tight_layout()
plt.show()

# --- After computing layer_stats and cv_attn ---

q_mean = layer_stats['mean_attn'].quantile(0.75)   # top 25% attention
#q_cv   = layer_stats['cv_attn'].quantile(0.50)     # below median variability

top_layers_attn = layer_stats[
    (layer_stats['mean_attn'] >= q_mean) #&
    #(layer_stats['cv_attn']   <= q_cv)
].sort_values('mean_attn', ascending=False)

print("Selected layers (high mean, low variability):")
print(top_layers_attn)


##################################################
###### LINEAR AND QUADRATIC MODEL FITS ###########
###### (WITH NORMALIZED RELEVANCE SCORE) #########
##################################################
from scipy import stats
import numpy as np


# --- Compute mean attention ratio (mid-level layers: 13–16) ---
selected_layers = top_layers_attn['layer'].tolist()
df_no_noise['mean_attn_ratio'] = df_no_noise['attn_ratio'].apply(
    #lambda x: np.mean(x[13:17]) if isinstance(x, (list, np.ndarray)) and len(x) > 16 else np.nan
    lambda x: np.mean([x[i] for i in selected_layers
                       if isinstance(x, (list, np.ndarray)) and i < len(x)])
)

import matplotlib.pyplot as plt

# --- Compute IQR bounds (mild trimming) ---
#Q1 = df_no_noise['mean_attn_ratio'].quantile(0.25)
#Q3 = df_no_noise['mean_attn_ratio'].quantile(0.75)
#IQR = Q3 - Q1
#lower = Q1 - 3 * IQR
#upper = Q3 + 3 * IQR
#
## --- Count and remove outliers ---
#n_before = len(df_no_noise)
#df_no_noise = df_no_noise[
#    (df_no_noise['mean_attn_ratio'] >= lower) &
#    (df_no_noise['mean_attn_ratio'] <= upper)
#].copy()
#n_after = len(df_no_noise)
#n_removed = n_before - n_after

# --- Plot histogram ---
#plt.figure(figsize=(6,4))
#plt.hist(df_no_noise['mean_attn_ratio'], bins=40, color='steelblue', edgecolor='black', alpha=0.7)
#plt.xlabel("Mean Attention Ratio")
#plt.ylabel("Frequency")
#plt.title("Distribution after Outlier Removal")
#plt.show()

#print(f"Removed {n_removed} outliers out of {n_before} samples ({n_removed / n_before:.2%}).")

# --- Remove missing values for reliable model fitting ---
valid_mask = df_no_noise['rel_score'].notna() & df_no_noise['mean_attn_ratio'].notna()
x_raw = df_no_noise.loc[valid_mask, 'rel_score'].values
y = df_no_noise.loc[valid_mask, 'mean_attn_ratio'].values

# --- Normalize rel_score to [0, 1] ---
x_min, x_max = x_raw.min(), x_raw.max()
x = (x_raw - x_min) / (x_max - x_min)

##################################################
# === FIT LINEAR AND QUADRATIC MODELS ===========
##################################################

# --- Linear fit: y = m·x + c ---
lin_coeffs = np.polyfit(x, y, 1)
lin_poly = np.poly1d(lin_coeffs)
y_pred_lin = lin_poly(x)

# --- Quadratic fit: y = a·x² + b·x + c ---
quad_coeffs = np.polyfit(x, y, 2)
quad_poly = np.poly1d(quad_coeffs)
y_pred_quad = quad_poly(x)

# --- Define R² computation ---
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

# --- Compute R² values ---
r2_lin = r2_score(y, y_pred_lin)
r2_quad = r2_score(y, y_pred_quad)

# --- Print model summaries ---
print("\n" + "="*70)
print("LINEAR AND QUADRATIC FITS (Normalized rel_score)")
print("="*70)
print(f"Linear:    y = {lin_coeffs[0]:.4f}·x + {lin_coeffs[1]:.4f}")
print(f"Quadratic: y = {quad_coeffs[0]:.4f}·x² + {quad_coeffs[1]:.4f}·x + {quad_coeffs[2]:.4f}")
print("-"*70)
print(f"R² (linear):    {r2_lin:.3f}")
print(f"R² (quadratic): {r2_quad:.3f}")
print("  Normalization ensures coefficients are scale-independent.")
print("  A higher R² indicates better model fit; large |a| implies curvature.")
print("="*70 + "\n")

##################################################
######### VISUALIZATION: NORMALIZED FITS #########
##################################################

# --- Generate smooth grid for fitted curves ---
x_grid = np.linspace(0, 1, 200)
y_grid_lin = lin_poly(x_grid)
y_grid_quad = quad_poly(x_grid)

plt.figure(figsize=(8, 6))

# --- Scatter: observed data ---
sns.scatterplot(
    x=x, y=y,
    alpha=0.45, s=60,
    edgecolor='white', linewidth=0.6,
    color='black',
    label='Samples'
)

# --- Linear fit ---
plt.plot(
    x_grid, y_grid_lin,
    color='steelblue', linestyle='--', linewidth=2,
    label=f'Linear fit ($R^2={r2_lin:.3f}$)'
)

# --- Quadratic fit ---
plt.plot(
    x_grid, y_grid_quad,
    color='crimson', linestyle='-', linewidth=3,
    label=f'Quadratic fit ($R^2={r2_quad:.3f}$)'
)

# --- Axes and figure styling ---
plt.title(
    "Attention Ratio vs Normalized Relatedness Score\n(Linear vs Quadratic Models)",
    fontsize=16, pad=15
)
plt.xlabel("Normalized Relatedness Score", fontsize=14)
plt.ylabel("Mean Attention Ratio", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(frameon=True, loc='best')
sns.despine()
plt.tight_layout()
plt.show()

##################################################
# SIGNIFICANCE TESTS FOR QUADRATIC (U-SHAPE)
##################################################
import statsmodels.api as sm
import statsmodels.formula.api as smf

# --- Regression dataframe ---
reg_df = pd.DataFrame({'y': y, 'x': x})

# --- Linear model: y ~ x ---
lin_mod = smf.ols('y ~ x', data=reg_df).fit()

# --- Quadratic model: y ~ x + x² ---
quad_mod = smf.ols('y ~ x + I(x**2)', data=reg_df).fit()

print("\n=== Linear model coefficients ===")
print(lin_mod.summary().tables[1])   # includes p-value for x

print("\n=== Quadratic model coefficients ===")
print(quad_mod.summary().tables[1])  # p-value for I(x**2) tests curvature

# --- Nested model comparison (does quadratic term help?) ---
anova_res = sm.stats.anova_lm(lin_mod, quad_mod)
print("\n=== Nested model comparison (linear vs quadratic) ===")
print(anova_res)

# --- Location of the minimum of the quadratic (on normalized x) ---
a = quad_mod.params['I(x ** 2)']
b = quad_mod.params['x']
x_vertex = -b / (2 * a)
print(f"\nQuadratic vertex (minimum) at x = {x_vertex:.3f} (normalized scale)")


# --- Goodness-of-fit metrics for both models ---

def model_metrics(mod, name):
    return {
        "model": name,
        "adj_R2":   mod.rsquared_adj,
        "RMSE":     np.sqrt(mod.mse_resid),  # √(residual mean square)
        "AIC":      mod.aic,
        "BIC":      mod.bic,                # optional, but often reported
        "df_model": int(mod.df_model + 1),  # number of parameters incl. intercept
    }

metrics_lin  = model_metrics(lin_mod,  "Linear (y ~ x)")
metrics_quad = model_metrics(quad_mod, "Quadratic (y ~ x + x²)")

metrics_df = pd.DataFrame([metrics_lin, metrics_quad])
print("\n=== Model comparison (goodness-of-fit) ===")
metrics_df


#%%
##################################################
# === COMPUTE ACCURACY METRICS (SOFT & HARD) ===
##################################################

# --- Expand the list column so each layer is a row ---
df['layer'] = df['attn_ratio'].apply(lambda x: list(range(len(x))))
df_exploded = df.explode(['attn_ratio', 'layer'])
df_exploded['attn_ratio'] = df_exploded['attn_ratio'].apply(pd.to_numeric, errors='coerce')

# --- Soft accuracy (based on cosine similarity threshold) ---
df_exploded['soft_accuracy'] = (df_exploded['long_caption_text_similarity_scores'] >= 0.9).astype(int)

# --- Clean model outputs and targets ---
df_exploded['output_clean'] = df_exploded['output_text'].str.replace(r'<\|im_end\|>', '', regex=True).str.replace(r'\.', '', regex=True).str.lower()
df_exploded['target_clean'] = df_exploded['target'].str.replace(r' \([^)]*\)', '', regex=True).str.lower()

# --- Compute Levenshtein-based hard accuracy ---
from Levenshtein import ratio
df_exploded['Levenshtein ratio'] = df_exploded.apply(lambda row: ratio(row['output_clean'], row['target_clean']), axis=1)
df_exploded['hard_accuracy'] = (df_exploded['Levenshtein ratio'] >= 0.55).astype(int)


#%%
##################################################
# === SPLIT DATASET BY ACCURACY AND COMPUTE STATS ===
##################################################

df_exploded_correct = df_exploded[df_exploded['soft_accuracy'] == 1]
df_exploded_wrong = df_exploded[df_exploded['soft_accuracy'] == 0]

print(df_exploded.shape[0])
print(df_exploded_correct.shape[0])
print(df_exploded_wrong.shape[0])

# --- Compute accuracy per condition (in %) ---
accuracy_per_condition = df_exploded.groupby(['Rel. Level', 'Noise Level', 'Noise Area']).agg(
    total_samples=('soft_accuracy', 'count'),
    correct_samples=('soft_accuracy', 'sum')
)
accuracy_per_condition['accuracy'] = (accuracy_per_condition['correct_samples'] /
                                      accuracy_per_condition['total_samples'] * 100).round(2)
print(accuracy_per_condition)

# --- Compute mean attention ratios by grouping conditions ---
grouped_means_complete = df_exploded.groupby(['Rel. Level', 'Noise Level', 'Noise Area', 'layer', 'soft_accuracy'])['attn_ratio'].mean().reset_index()
grouped_means = df_exploded.groupby(['Rel. Level', 'Noise Level', 'Noise Area', 'layer'])['attn_ratio'].mean().reset_index()
grouped_means_correct = df_exploded_correct.groupby(['Rel. Level', 'Noise Level', 'Noise Area', 'layer'])['attn_ratio'].mean().reset_index()
grouped_means_wrong = df_exploded_wrong.groupby(['Rel. Level', 'Noise Level', 'Noise Area', 'layer'])['attn_ratio'].mean().reset_index()

grouped_means['Dataset'] = 'COOCO'
grouped_means_correct['Dataset'] = 'COOCO'
grouped_means_wrong['Dataset'] = 'COOCO'

# --- Average across layers ---
grouped_layers = df_exploded.groupby(['Rel. Level', 'Noise Level', 'Noise Area'])['attn_ratio'].mean().reset_index()
grouped_layers_correct = df_exploded_correct.groupby(['Rel. Level', 'Noise Level', 'Noise Area'])['attn_ratio'].mean().reset_index()
grouped_layers_wrong = df_exploded_wrong.groupby(['Rel. Level', 'Noise Level', 'Noise Area'])['attn_ratio'].mean().reset_index()

# --- Merge all accuracy subsets ---
merged_layers = grouped_layers.merge(
    grouped_layers_correct, 
    on=['Noise Level', 'Rel. Level', 'Noise Area'], 
    suffixes=('_all', '_correct'),
    how='outer'
).merge(
    grouped_layers_wrong, 
    on=['Noise Level', 'Rel. Level', 'Noise Area'], 
    suffixes=('_correct', '_wrong'),
    how='outer'
)
merged_layers.rename(columns={'attn_ratio': 'attn_ratio_wrong'}, inplace=True)

# Export to LaTeX for paper-style visualization
print(merged_layers.round(3).to_latex(index=False))

##################################################
# === VISUALIZATION: ATTENTION RATIO BY LAYER ===
##################################################
grouped_means_visions = pd.read_csv('grouped_means_visions.csv')
grouped_means_correct_visions = pd.read_csv('grouped_means_correct_visions.csv')
grouped_means_incorrect_visions = pd.read_csv('grouped_means_incorrect_visions.csv')

grouped_means_visions['Dataset'] = 'VISIONS'
grouped_means_correct_visions['Dataset'] = 'VISIONS'
grouped_means_incorrect_visions['Dataset'] = 'VISIONS'

grouped_means_visions['Noise Area'] = (
    grouped_means_visions['Noise Area'].str.lower()
)
grouped_means_visions['Rel. Level'] = (
    grouped_means_visions['Rel. Level'].str.lower()
)
grouped_means_correct_visions['Noise Area'] = (
    grouped_means_correct_visions['Noise Area'].str.lower()
)
grouped_means_correct_visions['Rel. Level'] = (
    grouped_means_correct_visions['Rel. Level'].str.lower()
)
grouped_means_incorrect_visions['Noise Area'] = (
    grouped_means_incorrect_visions['Noise Area'].str.lower()
)
grouped_means_incorrect_visions['Rel. Level'] = (
    grouped_means_incorrect_visions['Rel. Level'].str.lower()
)

grouped_means_visions[(grouped_means_visions['Noise Area'] == 'target') & (grouped_means_visions['Noise Level'] == 0.0)]
toconcat1 = grouped_means_visions[(grouped_means_visions['Noise Area'] == 'target') & (grouped_means_visions['Noise Level'] == 0.0)]
toconcat2 = grouped_means_visions[(grouped_means_visions['Noise Area'] == 'target') & (grouped_means_visions['Noise Level'] == 0.0)]
toconcat1['Noise Area'] = 'context'
toconcat2['Noise Area'] = 'all'
grouped_means_visions = pd.concat([grouped_means_visions, toconcat1, toconcat2], ignore_index=True)

grouped_means_correct_visions[(grouped_means_correct_visions['Noise Area'] == 'target') & (grouped_means_correct_visions['Noise Level'] == 0.0)]
toconcat1 = grouped_means_correct_visions[(grouped_means_correct_visions['Noise Area'] == 'target') & (grouped_means_correct_visions['Noise Level'] == 0.0)]
toconcat2 = grouped_means_correct_visions[(grouped_means_correct_visions['Noise Area'] == 'target') & (grouped_means_correct_visions['Noise Level'] == 0.0)]
toconcat1['Noise Area'] = 'context'
toconcat2['Noise Area'] = 'all'
grouped_means_correct_visions = pd.concat([grouped_means_correct_visions, toconcat1, toconcat2], ignore_index=True)     

grouped_means_incorrect_visions[(grouped_means_incorrect_visions['Noise Area'] == 'target') & (grouped_means_incorrect_visions['Noise Level'] == 0.0)]
toconcat1 = grouped_means_incorrect_visions[(grouped_means_incorrect_visions['Noise Area'] == 'target') & (grouped_means_incorrect_visions['Noise Level'] == 0.0)]
toconcat2 = grouped_means_incorrect_visions[(grouped_means_incorrect_visions['Noise Area'] == 'target') & (grouped_means_incorrect_visions['Noise Level'] == 0.0)]
toconcat1['Noise Area'] = 'context'
toconcat2['Noise Area'] = 'all'
grouped_means_incorrect_visions = pd.concat([grouped_means_incorrect_visions, toconcat1, toconcat2], ignore_index=True)


grouped_means = pd.concat([grouped_means, grouped_means_visions], ignore_index=True)
grouped_means_correct = pd.concat([grouped_means_correct, grouped_means_correct_visions], ignore_index=True)
grouped_means_incorrect = pd.concat([grouped_means_wrong, grouped_means_incorrect_visions], ignore_index=True)

print(grouped_means['attn_ratio'][grouped_means['Noise Level'] == 0.0].mean())


#%%
grouped_means = grouped_means_correct

#%%
y_lim = 1.0
# Define the desired order of Rel. Level (edit this to your actual values)
REL_ORDER = ["low", "medium", "high", 'same target', 'original','incongruent', 'congruent']   # <-- put your own string levels here
order_map = {v: i for i, v in enumerate(REL_ORDER)}

def extract_rel_str(label: str) -> str:
    # from "Area: context, Rel: low" -> "low"
    return label.split("Rel:")[1].strip()

for noise_level_filter in [0.0, 0.5, 1.0]:
    filtered_data = grouped_means[grouped_means['Noise Level'] == noise_level_filter]
    plt.figure(figsize=(8, 6))

    for (condition, rel_level, dataset), sub_df in filtered_data.groupby(['Noise Area', 'Rel. Level', 'Dataset']):
        if noise_level_filter == 0.0 and condition not in ['target', 'Target']:
            continue
        sub_df = sub_df.sort_values(by='layer')

        if dataset == 'VISIONS':
            plt.plot(sub_df['layer'], sub_df['attn_ratio'],
                     marker='x', label=f'Dataset: VISIONS, Rel: {rel_level}')
        else:
            plt.plot(sub_df['layer'], sub_df['attn_ratio'],
                     marker='o', label=f'Dataset: COOCO, Rel: {rel_level}')

    plt.xlabel('Layer')
    plt.ylabel('Mean Attention Ratio')
    plt.ylim(0, y_lim)
    plt.title(f'Mean Attention Ratio per Layer (Noise = {noise_level_filter})')

    # --- legend sorted by Rel. Level (string) ---
    ax = plt.gca()
    lines, labels = ax.get_legend_handles_labels()

    sorted_pairs = sorted(
        zip(lines, labels),
        key=lambda x: order_map.get(extract_rel_str(x[1]), 999)
    )
    lines_sorted, labels_sorted = zip(*sorted_pairs)
    ax.legend(lines_sorted, labels_sorted)
    plt.grid()

plt.show()


#%%
##################################################
# === HEATMAPS: ATTENTION RATIO DISTRIBUTIONS ===
##################################################
noise_levels = [0.0, 0.5, 1.0]
conditions = ['all', 'context', 'target']

fig, axes = plt.subplots(len(noise_levels), len(conditions),
                         figsize=(15, 12), sharex=True, sharey=True)

for i, nl in enumerate(noise_levels):
    df_n = grouped_means[grouped_means['Noise Level'] == nl]
    for j, cond in enumerate(conditions):
        ax = axes[i, j]
        df_nc = df_n[df_n['Noise Area'] == cond]

        if df_nc.empty:
            ax.set_visible(False)
            continue

        pivot = df_nc.pivot_table(
            index='Rel. Level',
            columns='layer',
            values='attn_ratio',
            aggfunc='mean'  # or 'median', 'max', etc.
        )

        sns.heatmap(
            pivot,
            ax=ax,
            annot=False,
            vmin=0,
            vmax=1,
            linewidths=0.5,
            linecolor='gray',
            cbar=(j == len(conditions) - 1)
        )

        if i == 0:
            ax.set_title(cond.capitalize())
        if j == 0:
            ax.set_ylabel(f'Noise={nl}\nRelevance')
        ax.set_xlabel('Layer')

plt.tight_layout()
plt.show()


#%%
##################################################
# === ADVANCED HEATMAPS WITH DIVERGING COLORMAPS ===
##################################################
from matplotlib.colors import TwoSlopeNorm

# --- Define your desired order for Rel. Level strings ---
order_map = {v: i for i, v in enumerate(REL_ORDER)}

vmin, vmax, vcenter = 0, 1, 0.19
print(f"vmin: {vmin}, vcenter: {vcenter}, vmax: {vmax}")

fig, axes = plt.subplots(len(noise_levels), len(conditions),
                         figsize=(15, 12), sharex=True, sharey=True)

for i, nl in enumerate(noise_levels):
    df_n = grouped_means[grouped_means['Noise Level'] == nl]
    for j, cond in enumerate(conditions):
        ax = axes[i, j]
        df_nc = df_n[df_n['Noise Area'] == cond]

        if df_nc.empty:
            ax.set_visible(False)
            continue

        # Pivot table (handle duplicates)
        pivot = df_nc.pivot_table(
            index='Rel. Level',
            columns='layer',
            values='attn_ratio',
            aggfunc='mean'
        )

        # --- Sort rows (Rel. Level) by custom order ---
        pivot = pivot.loc[sorted(pivot.index, key=lambda x: order_map.get(x, 999))]

        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

        sns.heatmap(
            pivot,
            ax=ax,
            cmap="RdBu_r",
            norm=norm,
            linewidths=0.5,
            linecolor='gray',
            cbar=(j == len(conditions) - 1)
        )

        if i == 0:
            ax.set_title(cond.capitalize(), fontsize=20)
        if j == 0:
            ax.set_ylabel(f'Noise={nl}\nRelevance', fontsize=16)
        ax.set_xlabel('Layer', fontsize=16)
        ax.tick_params(labelsize=14)

plt.tight_layout()
plt.show()


#%%
##################################################
# === DELTA ANALYSIS: CORRECT vs WRONG ===
##################################################
merged = pd.merge(
    grouped_means_correct,
    grouped_means_wrong,
    on=['Rel. Level', 'Noise Level', 'Noise Area', 'layer'],
    suffixes=('_correct', '_wrong')
)
merged['attn_ratio_delta'] = merged['attn_ratio_correct'] - merged['attn_ratio_wrong']
merged['abs_delta'] = merged['attn_ratio_delta'].abs()

# Display top absolute deltas
top_deltas = merged.sort_values(by='abs_delta', ascending=False).head(10)
top_deltas


#%%
##################################################
# === VISUALIZE DELTAS (DIFFERENCES IN ATTENTION) ===
##################################################
vmin, vmax, vcenter = -0.25, 0.25, 0.0
print(f"vmin: {vmin}, vcenter: {vcenter}, vmax: {vmax}")

fig, axes = plt.subplots(len(noise_levels), len(conditions), figsize=(15, 12), sharex=True, sharey=True)

for i, nl in enumerate(noise_levels):
    df_n = merged[merged['Noise Level'] == nl]
    for j, cond in enumerate(conditions):
        ax = axes[i, j]
        df_nc = df_n[df_n['Noise Area'] == cond]
        pivot = df_nc.pivot(index='Rel. Level', columns='layer', values='attn_ratio_delta')
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        sns.heatmap(pivot, ax=ax, cmap="PiYG", norm=norm, linewidths=0.5, linecolor='gray', cbar=(j == len(conditions) - 1))
        if i == 0:
            ax.set_title(cond.capitalize(), fontsize=20)
        if j == 0:
            ax.set_ylabel(f'Noise={nl}\nRelevance', fontsize=16)
        ax.set_xlabel('Layer', fontsize=16)
        ax.tick_params(labelsize=14)

plt.tight_layout()
plt.show()
