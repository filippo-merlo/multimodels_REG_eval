#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Specify the folder containing your CSV files
folder_path = '/Users/filippomerlo/Desktop/output'

# Get a list of all CSV files in the specified folder
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# Initialize an empty list to store DataFrames
dataframes = []

# Loop through each CSV file and read it into a pandas DataFrame
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    dataframes.append(df)

# Concatenate all DataFrames into one combined DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)
df = combined_df.copy()

# Filter out rows where 'target' is "nothing"
df = df[df['target'] != "nothing"]

# Fill missing values in 'rel_level' with 'original'
df['rel_level'] = df['rel_level'].fillna('original')


# --- Filter dataset based on available image filenames ---
filtered_images_folder_path = '/Users/filippomerlo/Desktop/manually_filtered_images'

# Get all image filenames in the folder (only valid image formats)
image_filenames = {f for f in os.listdir(filtered_images_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))}

# Extract unique image IDs from filenames
image_filenames_id = {f.split('_')[0] for f in image_filenames}

# Ensure 'image_name' column exists before filtering
df = df[df['image_name'].apply(lambda x: x.split('_')[0] in image_filenames_id)] if 'image_name' in df.columns else df

#%% Filtering Data by Selected Models
# Define a list of model names to filter
#df['model_name'].unique()

desired_models = [
    'llava-hf/llava-onevision-qwen2-0.5b-si-hf',
]

desired_models = [
    'cyan2k/molmo-7B-O-bnb-4bit',
    'Salesforce/xgen-mm-phi3-mini-instruct-r-v1',
    'llava-hf/llava-onevision-qwen2-0.5b-si-hf',
    'Salesforce/xgen-mm-phi3-mini-instruct-singleimg-r-v1.5'
]


# Filter DataFrame to include only the selected models
subset_df = df[df['model_name'].isin(desired_models)]
df = subset_df.copy()

#%% Inspecting the DataFrame
# Display summary information about the dataset
df.info()
# Uncomment below lines for additional inspection
# df.sample(100)  # Randomly sample 100 rows
# df.describe()   # Show statistical summary
# df.head(10)     # Display the first 10 rows

#%% Semantic Similarity Analysis

# Group data by 'rel_level', 'noise_level', and 'condition', then compute mean scores
performance_by_combined = df.groupby(['rel_level', 'noise_level', 'condition'])[
    ['scores', 'text_similarity_scores', 'long_caption_scores', 'long_caption_text_similarity_scores']
].mean().reset_index()

# List of scores to visualize
scores = ['long_caption_scores', 'long_caption_text_similarity_scores']

# Generate line plots for each score metric
for score in scores:
    plt.figure(figsize=(14, 7))
    sns.lineplot(
        data=performance_by_combined,
        x='noise_level',
        y=score,
        hue='rel_level',
        style='condition',  # Differentiates between conditions
        markers=True,
        palette='tab10'
    )
    plt.title(f'{score.replace("_", " ").capitalize()} vs Noise Level by Condition and Rel Level')
    plt.xlabel('Noise Level')
    plt.ylabel(score.replace("_", " ").capitalize())
    plt.legend(title='Condition / Rel Level', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

#%% Accuracy Analysis
# Compute soft and hard accuracy metrics
df['soft_accuracy'] = (df['long_caption_text_similarity_scores'] > 0.9).astype(int)

from Levenshtein import ratio
# Compute similarity ratio between long_output and long_target
df['hard_accuracy'] = df.apply(lambda row: ratio(row['long_output'].lower(), row['long_target'].lower()) > 0.9, axis=1).astype(int)
#df['hard_accuracy'] = (
#    df['long_output'].str.lower().str.replace(' ', '').str.replace('/.', '') ==
#    df['long_target'].str.lower().str.replace(' ', '')
#).astype(int)

# Group by 'rel_level', 'noise_level', and 'condition' to compute mean accuracy
accuracy_by_combined = df.groupby(['rel_level', 'noise_level', 'condition'])[
    ['soft_accuracy', 'hard_accuracy']
].mean().reset_index()

# List of accuracy metrics to visualize
accuracies = ['soft_accuracy', 'hard_accuracy']

# Generate line plots for accuracy metrics
for accuracy in accuracies:
    plt.figure(figsize=(14, 7))
    sns.lineplot(
        data=accuracy_by_combined,
        x='noise_level',
        y=accuracy,
        hue='rel_level',
        style='condition',  # Differentiates between conditions
        markers=True,
        palette='tab10'
    )
    plt.title(f'{accuracy.replace("_", " ").capitalize()} vs Noise Level by Condition and Rel Level')
    plt.xlabel('Noise Level')
    plt.ylabel('Accuracy')
    plt.legend(title='Condition / Rel Level', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

