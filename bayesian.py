import numpy as np
import pandas as pd

# Load your dataset
data = pd.read_csv("Cleaned_Dataset.csv")

# Calculate Z-scores for the 'Arrival Delay' column
z_scores = np.abs((data['Arrival Delay'] - data['Arrival Delay'].mean()) / data['Arrival Delay'].std())

# Identify outliers (Z-score > 3)
outliers = z_scores > 3

# Remove outliers from the dataset
data_cleaned = data[~outliers]

# Save the cleaned dataset to a new CSV file
data_cleaned.to_csv("new_cleaned_dataset.csv", index=False)

print("Cleaned dataset saved as 'new_cleaned_dataset.csv'")
