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
# Step 2: Convert `scores` from tensor to numeric
df['scores'] = combined_df['scores'].str.extract(r"(\d+\.\d+)").astype(float)

# Display the combined DataFrame
df

#%%
import numpy as np

# Assuming the DataFrame is already loaded as `df`

# Step 1: Basic Information
print("Basic Info:")
print(df.info())
print("\nUnique Values per Column:")
print(df.nunique())

# Step 2: Convert `scores` from tensor to numeric
df['scores'] = df['scores'].str.extract(r"(\d+\.\d+)").astype(float)
#%%

# Step 3: Descriptive Statistics
print("\nDescriptive Statistics for Noise Levels and Scores:")
print(df[['noise_level', 'scores']].describe())

# Step 4: Count of Scenes and Models
scene_counts = df['scene'].value_counts()
model_counts = df['model_name'].value_counts()

print("\nTop Scenes:")
print(scene_counts.head())
print("\nTop Models:")
print(model_counts.head())

# Step 5: Agreement Analysis
df['is_correct'] = df['target'] == df['output']
agreement_rate = df['is_correct'].mean()
print(f"\nAgreement Rate: {agreement_rate:.2%}")

# Step 6: Scores by Correctness
mean_scores_correct = df[df['is_correct']]['scores'].mean()
mean_scores_incorrect = df[~df['is_correct']]['scores'].mean()

print(f"\nMean Score (Correct Predictions): {mean_scores_correct:.4f}")
print(f"Mean Score (Incorrect Predictions): {mean_scores_incorrect:.4f}")

# Optional Visualization Example
import matplotlib.pyplot as plt
import seaborn as sns

# Distribution of scores
sns.histplot(df['scores'], kde=True, bins=20)
plt.title("Score Distribution")
plt.xlabel("Scores")
plt.ylabel("Frequency")
plt.show()

# Bar plot for Scenes
scene_counts.plot(kind='bar', figsize=(10, 6), title="Scene Frequency")
plt.xlabel("Scene")
plt.ylabel("Count")
plt.show()

#%%
# Group by noise_level and calculate the mean score
scores_per_noise = df.groupby('noise_level')['scores'].mean()

# Print the results
print("Average Scores per Noise Level:")
print(scores_per_noise)

#%%
# Group by model_name and noise_level, then calculate the mean score
# Remove rows where target is "nothing"
filtered_df = df[df['target'] != "nothing"]
scores_per_model_noise = filtered_df.groupby(['model_name', 'noise_level'])['scores'].median()

# Print the results
print("Average Scores per Model per Noise Level:")
print(scores_per_model_noise)

#%%
filter = 0.45
# Filter rows where the score is greater than filter
filtered_scores = filtered_df[filtered_df['scores'] > filter]

# Group by model_name and noise_level and count the occurrences
frequency_per_model_noise = filtered_scores.groupby(['model_name', 'noise_level']).size()

# Print the frequency results
print(f"Frequency of models with score > {filter} per noise level:")
print(frequency_per_model_noise)

#%%
# Filter rows where the score is greater than 0.4
filtered_scores = filtered_df[filtered_df['scores'] > 0.45]

# Group by model_name, image_id, and noise_level, and count occurrences of scores > 0.4
image_score_counts_per_model = filtered_scores.groupby(['model_name', 'image_name', 'noise_level']).size().unstack(fill_value=0)

# Now, check if an image has a score > 0.4 for all three noise levels for each model
# This will return True if the image has a score > 0.4 for all three noise levels, False otherwise
appears_above_0_4_all_levels = (image_score_counts_per_model >= 1).sum(axis=1) == 3

# Group by model_name and count the number of images where the score is > 0.4 for all three levels
images_above_0_4_three_times = appears_above_0_4_all_levels.groupby('model_name').sum()

# Group by model_name and count the number of images where the score is not above 0.4 for all three levels
images_not_above_0_4_three_times = appears_above_0_4_all_levels.groupby('model_name').apply(lambda x: len(x) - x.sum())

# Print the results
print("Images with score > 0.4 for all three noise levels per model:")
print(images_above_0_4_three_times)

print("Images with score not > 0.4 for all three noise levels per model:")
print(images_not_above_0_4_three_times)
