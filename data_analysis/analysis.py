#%%
import os
import pandas as pd

# Specify the folder containing your CSV files
folder_path = '../outputs'

# Get a list of all CSV files in the folder
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# Initialize an empty list to store DataFrames
dataframes = []

# Loop through the list of CSV files and read them into pandas DataFrames
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    dataframes.append(df)


# Concatenate all DataFrames into one
combined_df = pd.concat(dataframes, ignore_index=True)
df = combined_df.copy()
df = df[df['target'] != "nothing"]

#%%

df.info()
df.describe()


#%%
# Group by model and noise level, then compute the mean of each score
performance_by_noise = df.groupby(['model_name', 'noise_level'])[
    ['scores', 'text_similarity_scores', 'long_caption_scores', 'long_caption_text_similarity_scores']
].mean().reset_index()

# Preview results
print(performance_by_noise)

# Optional: Plot the scores
import seaborn as sns
import matplotlib.pyplot as plt

scores = ['scores', 'text_similarity_scores', 'long_caption_scores', 'long_caption_text_similarity_scores']
for score in scores:
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=performance_by_noise, x='noise_level', y=score, hue='model_name')
    plt.title(f'{score.capitalize()} vs Noise Level')
    plt.xlabel('Noise Level')
    plt.ylabel(score)
    plt.legend(title='Model')
    plt.show()


#%%
# Drop rows with missing rel_level
df_rel = df.dropna(subset=['rel_level'])

# Group by rel_level and model_name, then compute mean scores
rel_level_performance = df_rel.groupby(['rel_level', 'model_name'])[
    ['scores', 'text_similarity_scores', 'long_caption_scores', 'long_caption_text_similarity_scores']
].mean().reset_index()

# Preview results
print(rel_level_performance)

# Optional: Plot the effect of rel_level
for score in scores:
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=rel_level_performance, x='rel_level', y=score, hue='model_name')
    plt.title(f'Effect of Relevance Level on {score.capitalize()}')
    plt.xlabel('Relevance Level')
    plt.ylabel(score)
    plt.legend(title='Model')
    plt.show()

#%%
import seaborn as sns
import matplotlib.pyplot as plt

# Drop rows where 'rel_level' is missing
df_rel = df.dropna(subset=['rel_level'])

# Group by model, noise level, and rel_level, and compute the mean of each score
interaction_performance = df_rel.groupby(['model_name', 'noise_level', 'rel_level'])[
    ['scores', 'text_similarity_scores', 'long_caption_scores', 'long_caption_text_similarity_scores']
].mean().reset_index()

# Preview the combined results
print(interaction_performance)

# Define the scores to analyze
scores = ['scores', 'text_similarity_scores', 'long_caption_scores', 'long_caption_text_similarity_scores']

# Plot interaction effects
for score in scores:
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=interaction_performance, 
        x='noise_level', 
        y=score, 
        hue='rel_level', 
        style='model_name', 
        markers=True
    )
    plt.title(f'{score.capitalize()} vs Noise Level by Relevance Level and Model')
    plt.xlabel('Noise Level')
    plt.ylabel(score)
    plt.legend(title='Relevance Level & Model')
    plt.tight_layout()
    plt.show()


#%% STATISTICAL ANALYSIS

# effect of noyse level
from scipy.stats import f_oneway

# effect of rel level
print('Effect of rel level\n')
# Perform ANOVA for each score across rel_levels for a specific model
for model in df_rel['model_name'].unique():
    print(f"Model: {model}")
    model_data = df_rel[df_rel['model_name'] == model]
    for score in scores:
        groups = [group[score].values for _, group in model_data.groupby('rel_level')]
        f_stat, p_value = f_oneway(*groups)
        print(f"{score}: F-statistic={f_stat:.4f}, p-value={p_value:.4e}")
    print("\n")
