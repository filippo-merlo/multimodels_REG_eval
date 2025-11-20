#%%
##################################################
########### IMPORTS & INITIAL CONFIGURATION ###########
##################################################
import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import os
from pprint import pprint

tqdm.pandas()

# --- Matplotlib visual style setup ---
plt.rcParams.update({
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.title_fontsize': 14,
    'legend.fontsize': 12,
    'font.size': 14
})

separator = "\n\n##################################################\n##################################################\n\n"


#%%
##################################################
################### LOAD DATA ####################
##################################################
dataset_path = '/home/fmerlo/data/sceneregstorage/attn_eval_output/results_att_deployment_VISIONS_complete.csv'

df = pd.read_csv(dataset_path)
df['Noise Area'] = df['Noise Area'].fillna('Target')


#%%
##################################################
########### PARSE LIST-LIKE COLUMNS ##############
##################################################
def parse_list(value):
    """Convert string representations of lists into actual lists."""
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return []
    return value


##################################################
########### COMPUTE ATTENTION RATIOS #############
##################################################
def compute_ratio(row):
    """Compute ratio of attention over target to context per layer."""
    attn_over_target = np.array(row['attn_over_target'], dtype=np.float32)
    attn_over_context = np.array(row['attn_over_context'], dtype=np.float32)
    ratio = np.divide(attn_over_target, attn_over_context, out=np.full_like(attn_over_target, np.nan), where=attn_over_context!=0)
    return ratio.tolist()

df['attn_over_target'] = df['attn_over_target'].apply(parse_list)
df['attn_over_context'] = df['attn_over_context'].apply(parse_list)
df['attn_ratio'] = df.progress_apply(compute_ratio, axis=1)


#%%
##################################################
################# ACCURACY SETUP #################
##################################################
# --- Compute soft accuracy ---
df['soft_accuracy'] = (df['long_caption_text_similarity_score'] >= 0.9).astype(int)
df['soft_accuracy_modal_names'] = (df['output_modal_name_text_similarity_scores'] >= 0.9).astype(int)

##################################################
########### CORRELATION & DIVERGENCE #############
##################################################

# --- Compute Pearson correlation ---
corr = df[['soft_accuracy', 'soft_accuracy_modal_names']].corr().iloc[0, 1]

# --- Compute divergence cases ---
divergent = df[df['soft_accuracy'] != df['soft_accuracy_modal_names']]
n_divergent = len(divergent)
divergence_rate = n_divergent / len(df)

# --- Optional: show contingency table for deeper inspection ---
contingency = pd.crosstab(df['soft_accuracy'], df['soft_accuracy_modal_names'],
                          rownames=['soft_accuracy'], colnames=['soft_accuracy_modal_names'])
# --- Pretty print results ---
print("\n" + "="*60)
print("ACCURACY CORRELATION AND DIVERGENCE ANALYSIS")
print("="*60)
print(f"Pearson correlation:       {corr:.3f}")
print(f"Divergent cases:           {n_divergent} / {len(df)}  ({divergence_rate*100:.2f}%)")
print("-"*60)
print("Contingency table (counts):")
print(contingency.to_string())
print("-"*60)

# --- Add normalized version for proportions ---
contingency_norm = contingency.div(contingency.sum(axis=1), axis=0).round(3) * 100
print("Contingency table (row %):")
print(contingency_norm.to_string())
print("="*60 + "\n")

#%%
##################################################
##### LAYER-WISE ATTENTION VS CONSISTENCY ########
##################################################

# --- Subset: no-noise condition only ---
df_no_noise = df[df['Noise Level'] == 0.0].copy()

# --- Filter and keep only correct samples ---
df_no_noise = df_no_noise[df_no_noise['soft_accuracy'] == 1].copy()

# --- Parse 'object.consistency' into numeric mean and std ---
df_no_noise[['obj_consistency_mean', 'obj_consistency_std']] = (
    df_no_noise['object.consistency']
    .str.extract(r'([\d\.]+)\s*±\s*([\d\.]+)')
    .astype(float)
)

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
###### NON-LINEAR (U-SHAPE) / QUADRATIC ANALYSIS ##
###### WITH NORMALIZED CONSISTENCY SCORE ##########
##################################################

# --- Subset: no-noise condition only ---
df_no_noise = df[df['Noise Level'] == 0.0].copy()

# --- Filter and keep only correct samples ---
df_no_noise = df_no_noise[df_no_noise['soft_accuracy'] == 1].copy()

# --- Compute mean attention ratios (mid-level layers) ---
selected_layers = top_layers_attn['layer'].tolist()
df_no_noise['mean_attn_ratio'] = df_no_noise['attn_ratio'].apply(
    #lambda x: np.mean(x[13:17]) if isinstance(x, (list, np.ndarray)) and len(x) > 16 else np.nan
    lambda x: np.mean([x[i] for i in selected_layers
                       if isinstance(x, (list, np.ndarray)) and i < len(x)])
)

# --- Parse 'object.consistency' field (extract mean and std) ---
df_no_noise[['obj_consistency_mean', 'obj_consistency_std']] = (
    df_no_noise['object.consistency']
    .str.extract(r'([\d\.]+)\s*±\s*([\d\.]+)')
    .astype(float)
)

import matplotlib.pyplot as plt

## --- Compute IQR bounds (mild trimming) ---
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
plt.figure(figsize=(6,4))
plt.hist(df_no_noise['mean_attn_ratio'], bins=40, color='steelblue', edgecolor='black', alpha=0.7)
plt.xlabel("Mean Attention Ratio")
plt.ylabel("Frequency")
plt.title("Distribution after Outlier Removal")
plt.show()

#print(f"Removed {n_removed} outliers out of {n_before} samples ({n_removed / n_before:.2%}).")

# --- Check parsing ---
print("Parsed object consistency columns:\n",
      df_no_noise[['object.consistency', 'obj_consistency_mean', 'obj_consistency_std']].head())


# --- Drop NaNs for safe fitting ---
valid_mask = df_no_noise['obj_consistency_mean'].notna() & df_no_noise['mean_attn_ratio'].notna()
x_raw = df_no_noise.loc[valid_mask, 'obj_consistency_mean'].values
y = df_no_noise.loc[valid_mask, 'mean_attn_ratio'].values
print(y.max())

# --- Normalize object consistency to [0, 1] ---
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

# --- Compute R² values ---
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

r2_lin = r2_score(y, y_pred_lin)
r2_quad = r2_score(y, y_pred_quad)

# --- Print model summaries ---
print("\n" + "="*60)
print("LINEAR AND QUADRATIC FITS (Normalized Object Consistency)")
print("="*60)
print(f"Linear:    y = {lin_coeffs[0]:.4f}·x + {lin_coeffs[1]:.4f}")
print(f"Quadratic: y = {quad_coeffs[0]:.4f}·x² + {quad_coeffs[1]:.4f}·x + {quad_coeffs[2]:.4f}")
print("-"*60)
print(f"R² (linear):    {r2_lin:.3f}")
print(f"R² (quadratic): {r2_quad:.3f}")
print("  Normalization ensures scale-independent comparison.")
print("  A U-shaped trend should yield a higher R² and large |a| in the quadratic term.")
print("="*60 + "\n")

##################################################
######### COMBINED LINEAR + QUADRATIC PLOT ########
##################################################

# --- Prepare smooth grid for curves ---
x_grid = np.linspace(0, 1, 200)
y_grid_lin = lin_poly(x_grid)
y_grid_quad = quad_poly(x_grid)

plt.figure(figsize=(8, 6))

# --- Scatter: observed data ---
sns.scatterplot(
    x=x, y=y,
    alpha=0.45, s=60,
    edgecolor='white', linewidth=0.6,
    color='royalblue',
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
    color='crimson', linewidth=3,
    label=f'Quadratic fit ($R^2={r2_quad:.3f}$)'
)

# --- Styling ---
plt.title(
    "Attention Ratio vs. Normalized Object Consistency\n(Linear vs Quadratic Models)",
    fontsize=16, pad=15
)
plt.xlabel("Normalized Object Consistency", fontsize=14)
plt.ylabel("Mean Attention Ratio", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(frameon=True, loc='best')
sns.despine()
plt.tight_layout()
plt.show()

#%%
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


#%%
##################################################
########### SPLIT: CORRECT vs WRONG ##############
##################################################

# --- Expand per-layer data ---
df['layer'] = df['attn_ratio'].apply(lambda x: list(range(len(x))))
df_exploded = df.explode(['attn_ratio', 'layer'])
df_exploded['attn_ratio'] = df_exploded['attn_ratio'].apply(pd.to_numeric, errors='coerce')


df_exploded_correct = df_exploded[df_exploded['soft_accuracy'] == 1]
df_exploded_wrong = df_exploded[df_exploded['soft_accuracy'] == 0]

print(df_exploded.shape[0])
print(df_exploded_correct.shape[0])
print(df_exploded_wrong.shape[0])

# --- Compute accuracy per condition ---
accuracy_per_condition = df_exploded.groupby(['Rel. Level', 'Noise Level', 'Noise Area']).agg(
    total_samples=('soft_accuracy', 'count'),
    correct_samples=('soft_accuracy', 'sum')
)
accuracy_per_condition['accuracy'] = (accuracy_per_condition['correct_samples'] / accuracy_per_condition['total_samples'] * 100).round(2)
print(accuracy_per_condition)


##################################################
########### GROUPING & MERGING STATS #############
##################################################
grouped_means_complete = df_exploded.groupby(['Rel. Level', 'Noise Level', 'Noise Area', 'layer','soft_accuracy'])['attn_ratio'].mean().reset_index()
grouped_means = df_exploded.groupby(['Rel. Level', 'Noise Level', 'Noise Area', 'layer'])['attn_ratio'].mean().reset_index()
grouped_means_correct = df_exploded_correct.groupby(['Rel. Level', 'Noise Level', 'Noise Area', 'layer'])['attn_ratio'].mean().reset_index()
grouped_means_wrong = df_exploded_wrong.groupby(['Rel. Level', 'Noise Level', 'Noise Area', 'layer'])['attn_ratio'].mean().reset_index()

grouped_layers = df_exploded.groupby(['Rel. Level', 'Noise Level', 'Noise Area'])['attn_ratio'].mean().reset_index()
grouped_layers_correct = df_exploded_correct.groupby(['Rel. Level', 'Noise Level', 'Noise Area'])['attn_ratio'].mean().reset_index()
grouped_layers_wrong = df_exploded_wrong.groupby(['Rel. Level', 'Noise Level', 'Noise Area'])['attn_ratio'].mean().reset_index()

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
accuracy_per_condition


#%%
##################################################
################## LATEX TABLE ###################
##################################################
print(merged_layers.round(3).to_latex(index=False))


#%%
##################################################
############### LINE PLOT SECTION ################
##################################################
grouped_means = grouped_means_wrong
y_lim = 1
grouped_means.to_csv("grouped_means_incorrect_visions.csv", index=False)

# --- Noise = 0.0 ---
noise_level_filter = 0.0
filtered_data = grouped_means[grouped_means['Noise Level'] == noise_level_filter]

#%%

plt.figure(figsize=(8, 6))
for (condition, rel_level), sub_df in filtered_data.groupby(['Noise Area', 'Rel. Level']):
    if condition != 'Target':
        continue
    plt.plot(sub_df.sort_values('layer')['layer'], sub_df['attn_ratio'], marker='o', label=f'Rel: {rel_level}')
plt.xlabel('Layer'); plt.ylabel('Mean Attention Ratio')
plt.ylim(0, y_lim)
plt.title(f'Mean Attention Ratio per Layer (Noise = {noise_level_filter})')
plt.legend(); plt.grid(); plt.show()

# --- Noise = 0.5 ---
noise_level_filter = 0.5
filtered_data = grouped_means[grouped_means['Noise Level'] == noise_level_filter]
plt.figure(figsize=(8, 6))
for (condition, rel_level), sub_df in filtered_data.groupby(['Noise Area', 'Rel. Level']):
    plt.plot(sub_df.sort_values('layer')['layer'], sub_df['attn_ratio'], marker='o', label=f'Area: {condition}, Rel: {rel_level}')
plt.xlabel('Layer'); plt.ylabel('Mean Attention Ratio')
plt.ylim(0, y_lim)
plt.title(f'Mean Attention Ratio per Layer (Noise = {noise_level_filter})')
plt.legend(); plt.grid(); plt.show()

# --- Noise = 1.0 ---
noise_level_filter = 1.0
filtered_data = grouped_means[grouped_means['Noise Level'] == noise_level_filter]
plt.figure(figsize=(8, 6))
for (condition, rel_level), sub_df in filtered_data.groupby(['Noise Area', 'Rel. Level']):
    plt.plot(sub_df.sort_values('layer')['layer'], sub_df['attn_ratio'], marker='o', label=f'Area: {condition}, Rel: {rel_level}')
plt.xlabel('Layer'); plt.ylabel('Mean Attention Ratio')
plt.ylim(0, y_lim)
plt.title(f'Mean Attention Ratio per Layer (Noise = {noise_level_filter})')
plt.legend(); plt.grid(); plt.show()


#%%
##################################################
################# HEATMAP SECTION ################
##################################################
noise_levels = [0.0, 0.5, 1.0]
conditions = ['All', 'Context', 'Target']
n_rows, n_cols = len(noise_levels), len(conditions)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), sharex=True, sharey=True)
for i, nl in enumerate(noise_levels):
    df_n = grouped_means[grouped_means['Noise Level'] == nl]
    for j, cond in enumerate(conditions):
        ax = axes[i, j]
        df_nc = df_n[df_n['Noise Area'] == cond]
        if nl == 0.0:
            df_nc = df_n[df_n['Noise Area'] == 'Target']
        pivot = df_nc.pivot(index='Rel. Level', columns='layer', values='attn_ratio')
        sns.heatmap(pivot, ax=ax, annot=False, fmt=".2f", vmin=0, vmax=1,
                    linewidths=0.5, linecolor='gray', cbar=(j == n_cols - 1))
        if i == 0: ax.set_title(cond.capitalize())
        if j == 0: ax.set_ylabel(f'Noise={nl}\nRelevance')
        ax.set_xlabel('Layer')
plt.tight_layout(); plt.show()


#%%
##################################################
######## DIVERGING HEATMAP WITH CENTERING ########
##################################################
from matplotlib.colors import TwoSlopeNorm
import matplotlib.gridspec as gridspec

all_vals = grouped_means['attn_ratio']
vmin, vmax = [0, 1]
vcenter = grouped_means[grouped_means['Noise Level'] == 0.0]['attn_ratio'].mean()
vcenter = 0.15
print(f"vmin: {vmin}, vcenter: {vcenter}, vmax: {vmax}")


#%%
##################################################
######## FINAL HEATMAP GRID (COLOR NORMALIZED) ###
##################################################
noise_levels = [0.0, 0.5, 1.0]
conditions = ['All', 'Context', 'Target']
n_rows, n_cols = len(noise_levels), len(conditions)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), sharex=True, sharey=True)
for i, nl in enumerate(noise_levels):
    df_n = grouped_means[grouped_means['Noise Level'] == nl]
    for j, cond in enumerate(conditions):
        ax = axes[i, j]
        df_nc = df_n[df_n['Noise Area'] == cond]
        if nl == 0.0: df_nc = df_n[df_n['Noise Area'] == 'Target']
        pivot = df_nc.pivot(index='Rel. Level', columns='layer', values='attn_ratio')
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        sns.heatmap(pivot, ax=ax, cmap="RdBu_r", norm=norm,
                    linewidths=0.5, linecolor='gray', cbar=(j == n_cols - 1))
        if i == 0: ax.set_title(cond.capitalize(), fontsize=20)
        if j == 0: ax.set_ylabel(f'Noise={nl}\nRelevance', fontsize=16)
        ax.set_xlabel('Layer', fontsize=16)
        ax.tick_params(labelsize=14)
plt.tight_layout(); plt.show()


#%%
##################################################
################ DELTA COMPUTATION ################
##################################################
merged = pd.merge(
    grouped_means_correct, grouped_means_wrong,
    on=['Rel. Level', 'Noise Level', 'Noise Area', 'layer'],
    suffixes=('_correct', '_wrong')
)
merged['attn_ratio_delta'] = merged['attn_ratio_correct'] - merged['attn_ratio_wrong']
merged['abs_delta'] = merged['attn_ratio_delta'].abs()
top_deltas = merged.sort_values(by='abs_delta', ascending=False).head(10)
top_deltas


#%%
##################################################
############## DELTA HEATMAP PLOT ################
##################################################
vmin, vmax, vcenter = -0.35, 0.35, 0.0
print(f"vmin: {vmin}, vcenter: {vcenter}, vmax: {vmax}")

noise_levels = [0.0, 0.5, 1.0]
conditions = ['All', 'Context', 'Target']
n_rows, n_cols = len(noise_levels), len(conditions)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), sharex=True, sharey=True)
for i, nl in enumerate(noise_levels):
    df_n = merged[merged['Noise Level'] == nl]
    for j, cond in enumerate(conditions):
        ax = axes[i, j]
        df_nc = df_n[df_n['Noise Area'] == cond]
        if nl == 0.0: df_nc = df_n[df_n['Noise Area'] == 'Target']
        pivot = df_nc.pivot(index='Rel. Level', columns='layer', values='attn_ratio_delta')
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        sns.heatmap(pivot, ax=ax, cmap="PiYG", norm=norm,
                    linewidths=0.5, linecolor='gray', cbar=(j == n_cols - 1))
        if i == 0: ax.set_title(cond.capitalize(), fontsize=20)
        if j == 0: ax.set_ylabel(f'Noise={nl}\nRelevance', fontsize=16)
        ax.set_xlabel('Layer', fontsize=16)
plt.tight_layout(); plt.show()
