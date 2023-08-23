import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file, skipping the first 14 lines and only loading the first column
df = pd.read_csv('data.csv', skiprows=41, usecols=[0], header=None)

# Define the start and end cells for each condition
conditions = {
    'Control': ('Control Beginning', 'Control End'),
    'Condition 1': ('Trigger Condition 1 Start', 'Trigger Condition 1 End'),
    'Condition 2': ('Trigger Condition 2 Start', 'Trigger Condition 2 End'),
    'Condition 3': ('Trigger Condition 3 Start', 'Trigger Condition 3 End'),
}

# Analyze each condition
for condition, (start, end) in conditions.items():
    # Check if the start and end markers exist
    if start in df[0].values and end in df[0].values:
        # Find the indices of the start and end markers
        start_index = df[df[0] == start].index[0]
        end_index = df[df[0] == end].index[0]

        # Extract the data for this condition
        condition_data = df.loc[start_index+1:end_index-1, 0]

        # Convert the data to numeric, ignoring any errors
        condition_data = pd.to_numeric(condition_data, errors='coerce')

        # Drop any NaN values
        condition_data = condition_data.dropna()

        # Perform basic data analysis
        print(f'--- {condition} ---')
        print(condition_data.describe())

        # Plot a histogram
        plt.hist(condition_data, bins=20, alpha=0.5)
        plt.title(f'Histogram of {condition}')
        plt.show()
    else:
        print(f'Start or end marker for {condition} not found in data.')
