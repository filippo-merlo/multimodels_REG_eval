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
scores_per_model_noise = filtered_df.groupby(['model_name', 'noise_level'])['scores'].mean()

# Print the results
print("Average Scores per Model per Noise Level:")
print(scores_per_model_noise)