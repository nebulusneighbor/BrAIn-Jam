import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = '/data.csv'

# Read the file line by line and replace commas inside quotes with a placeholder
lines = []
with open(file_path, 'r') as file:
    for line in file:
        inside_quotes = False
        new_line = ''
        for char in line:
            if char == '"':
                inside_quotes = not inside_quotes
            if char == ',' and inside_quotes:
                new_line += ';'  # Replace commas inside quotes with semicolon
            else:
                new_line += char
        lines.append(new_line)

# Convert the lines to a single string
csv_content = '\n'.join(lines)

# Read the content as a CSV
csv_file = StringIO(csv_content)
df = pd.read_csv(csv_file, sep=',')

# Replace the semicolons back to commas in the "Trigger" column
df['Trigger'] = df['Trigger'].str.replace(';', ',')

# Remove duplicate triggers
df['Trigger'] = df['Trigger'].where(df['Trigger'] != df['Trigger'].shift(1))

# Define the conditions for analysis
conditions = {
    'Control': ('Control Beginning', 'Control End'),
    'Condition 1': ('Trigger Condition 1 Start', 'Trigger Condition 1 End'),
    'Condition 2': ('Trigger Condition 2 Start', 'Trigger Condition 2 End'),
    'Condition 3': ('Trigger Condition 3 Start', 'Trigger Condition 3 End'),
}

# Define regions
regions = ['Frontopolar', 'DLPFC_Left', 'DLPFC_Right', 'MPFC', 'TPJ_Left', 'TPJ_Right']
labels = ['HbO', 'HbR', 'SlidingOverallHbO', 'SlidingOverallHbR']

# Analyze each condition for each region and label
for condition, (start, end) in conditions.items():
    for region in regions:
        for label in labels:
            # Filter the data for this condition, region, and label
            condition_data = df[(df['Trigger'] == start) | (df['Trigger'] == end) | (df['Type'].str.contains(f'{region}_{label}'))]

            # Find the indices of the start and end markers
            start_index = condition_data[condition_data['Trigger'] == start].index[0]
            end_index = condition_data[condition_data['Trigger'] == end].index[-1]

            # Extract the data for this condition, region, and label
            condition_values = condition_data.loc[start_index + 1:end_index - 1, 'Value']

            # Convert the data to numeric, ignoring any errors
            condition_values = pd.to_numeric(condition_values, errors='coerce')

            # Drop any NaN values
            condition_values = condition_values.dropna()

            # Perform basic data analysis
            print(f'--- {condition} - {region} - {label} ---')
            print(condition_values.describe())

            # Plot a histogram
            plt.hist(condition_values, bins=20, alpha=0.5)
            plt.title(f'Histogram of {condition} - {region} - {label}')
            plt.show()
