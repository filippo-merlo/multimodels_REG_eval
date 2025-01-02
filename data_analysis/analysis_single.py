#%%
import os
import pandas as pd

# Specify the folder containing your CSV files
folder_path = '/Users/filippomerlo/Desktop/output'

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

#%%
df['model_name'].unique()
#%%
model_names = ['cyan2k/molmo-7B-O-bnb-4bit',
       'Salesforce/xgen-mm-phi3-mini-instruct-r-v1',
       'llava-hf/llava-onevision-qwen2-0.5b-si-hf',
       'Salesforce/xgen-mm-phi3-mini-instruct-singleimg-r-v1.5']
subset_df = df[df['model_name'].isin(model_names)]
subset_df.head(10)
df = subset_df.copy()
#%% Insoect
#df.sample(100)
df.info()
#df.describe()
#df.head(10)

#%% Semantic similarity
import matplotlib.pyplot as plt
import seaborn as sns


# Merge condition and rel_level into a single categorical column for grouping
#df['condition_rel_level'] = df['condition'] + "_" + df['rel_level'].astype(str)

# Group by model, noise level, and the combined column, then compute the mean of each score
performance_by_combined = df.groupby(['rel_level', 'noise_level', 'condition'])[
    ['scores', 'text_similarity_scores', 'long_caption_scores', 'long_caption_text_similarity_scores']
].mean().reset_index()


# Plot scores
scores = ['long_caption_scores', 'long_caption_text_similarity_scores']
for score in scores:
    plt.figure(figsize=(14, 7))
    sns.lineplot(
        data=performance_by_combined,
        x='noise_level',
        y=score,
        hue='rel_level',
        style='condition',  # Style differentiates the combined condition_rel_level
        markers=True,
        palette='tab10'
    )
    plt.title(f'{score.capitalize()} vs Noise Level by Condition and Rel Level')
    plt.xlabel('Noise Level')
    plt.ylabel(score)
    plt.legend(title='Model / Condition-Rel Level', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

#%% Accuracy

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming the DataFrame is loaded as df
# Add soft and hard accuracy columns
df['soft_accuracy'] = (df['long_caption_text_similarity_scores'] > 0.9).astype(int)
df['hard_accuracy'] = (df['long_output'].str.lower().replace(' ', '').replace('/.', '') == df['long_target'].str.lower().replace(' ', '')).astype(int)

#%%
# Group by model, noise level, condition, and rel_level

grouped = df.groupby(['rel_level', 'noise_level', 'condition'])

# Compute the mean for soft and hard accuracy
accuracy_by_combined = grouped[['soft_accuracy', 'hard_accuracy']].mean().reset_index()

# Plot soft and hard accuracy
accuracies = ['soft_accuracy', 'hard_accuracy']
for accuracy in accuracies:
    plt.figure(figsize=(14, 7))
    sns.lineplot(
        data=accuracy_by_combined,
        x='noise_level',
        y=accuracy,
        hue='rel_level',
        style='condition',  # Style differentiates the combined condition_rel_level
        markers=True,
        palette='tab10'
    )
    plt.title(f'{accuracy.replace("_", " ").capitalize()} vs Noise Level by Condition and Rel Level')
    plt.xlabel('Noise Level')
    plt.ylabel('Accuracy')
    plt.legend(title='Condition / Rel Level', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

