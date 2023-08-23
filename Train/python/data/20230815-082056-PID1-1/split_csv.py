import pandas as pd

# Read the original CSV file
df = pd.read_csv('data_clean.csv')

# Make sure the 'Timestamp' column is a datetime object for proper sorting
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Filter out rows where 'Type Mean' and the last 'Value' column are NaN
df1_a = df.dropna(subset=['Type Mean', df.columns[5]])

# Filter rows where 'Trigger' column is not null
df1_b = df[df['Trigger'].notna()]

# Concatenate the two filtered DataFrames
df1 = pd.concat([df1_a, df1_b], ignore_index=True)

# Select columns for the first CSV: Timestamp, Trigger, 'Type Mean', and the last 'Value' column (index 5)
df1 = df1[['Timestamp', 'Trigger', 'Type Mean', df.columns[5]]]

# Sort by 'Timestamp' to make it chronological
df1 = df1.sort_values(by='Timestamp')

# Write the selected columns to the first CSV file
df1.to_csv('Sliding_Avg.csv', index=False)

# For the second CSV, filter out rows where both 'Type' and the 'Value' column (index 3) are missing
df2 = df.dropna(subset=['Type', df.columns[3]], how='all')

# Select columns for the second CSV: Timestamp, Trigger, Type, and the 'Value' column (index 3)
df2 = df2[['Timestamp', 'Trigger', 'Type', df.columns[3]]]

# Write the selected columns to the second CSV file
df2.to_csv('HbO_HbR.csv', index=False)
