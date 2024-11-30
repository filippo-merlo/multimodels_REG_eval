#%%
import os
import pandas as pd

# Specify the folder containing your CSV files
folder_path = '/Users/filippomerlo/Desktop/outputs'

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
df['rel_level'] = df['rel_level'].fillna('original')

#%% Insoect
df.sample(100)
df.info()
df.describe()
df.head(10)

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


#%% Accuracy

# Assuming the DataFrame is named `df`
df['rel_level'] = df['rel_level'].fillna('Unknown')  # Handle NaN rel_levels

# Grouping and calculations
grouped = df.groupby(['model_name', 'rel_level', 'noise_level'])

# Count total entries in each group
total_count = grouped.size().reset_index(name='total_count')

# Count "perfect" scores
perfect_count = df[df['text_similarity_scores'] > 0.98].groupby(['model_name', 'rel_level', 'noise_level']).size().reset_index(name='perfect_count')

# Count "soft" scores
soft_count = df[df['text_similarity_scores'] > 0.9].groupby(['model_name', 'rel_level', 'noise_level']).size().reset_index(name='soft_count')

# Merge counts
accuracy = total_count.merge(perfect_count, on=['model_name', 'rel_level', 'noise_level'], how='left') \
                      .merge(soft_count, on=['model_name', 'rel_level', 'noise_level'], how='left')

# Fill NaN values with 0 (in case some groups have no "perfect" or "soft" scores)
accuracy.fillna(0, inplace=True)

# Calculate accuracy percentages
accuracy['perfect_accuracy'] = accuracy['perfect_count'] / accuracy['total_count']
accuracy['soft_accuracy'] = accuracy['soft_count'] / accuracy['total_count']

# Display the result
accuracy

import matplotlib.pyplot as plt
import seaborn as sns

# Convert `noise_level` to a categorical variable for better plotting
accuracy['noise_level'] = accuracy['noise_level'].astype(str)

# Melt the DataFrame for easier plotting
accuracy_melted = accuracy.melt(
    id_vars=['model_name', 'rel_level', 'noise_level'],
    value_vars=['perfect_accuracy', 'soft_accuracy'],
    var_name='Accuracy Type',
    value_name='Accuracy'
)

# Plot
plt.figure(figsize=(15, 8))
sns.lineplot(
    data=accuracy,
    x='noise_level',
    y='perfect_accuracy',
    hue='model_name',
    style='rel_level',
    markers=True,
    dashes=False,
    palette='tab10'
)
plt.title('Perfect Accuracy by Noise Level', fontsize=16)
plt.xlabel('Noise Level', fontsize=14)
plt.ylabel('Perfect Accuracy', fontsize=14)
plt.legend(title='Model and Rel Level', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot
plt.figure(figsize=(15, 8))
sns.lineplot(
    data=accuracy,
    x='noise_level',
    y='soft_accuracy',
    hue='model_name',
    style='rel_level',
    markers=True,
    dashes=False,
    palette='tab10'
)
plt.title('Soft Accuracy by Noise Level', fontsize=16)
plt.xlabel('Noise Level', fontsize=14)
plt.ylabel('Perfect Accuracy', fontsize=14)
plt.legend(title='Model and Rel Level', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()
