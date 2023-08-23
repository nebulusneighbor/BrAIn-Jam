import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('data_clean_2x.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Filter out rows with average values
df = df[~df['Type'].isin(['SlidingOverallHbO', 'SlidingOverallHbR'])]
# Initialize variables
conditions_corrected = ['Control', 'Condition 1', 'Condition 2', 'Condition 3']
time_series_data_corrected = {condition: {'HbO': [], 'HbR': []} for condition in conditions_corrected}
average_data_corrected = {condition: {'HbO': 0.0, 'HbR': 0.0} for condition in conditions_corrected}
current_condition_corrected = None

# Iterate through rows to identify conditions and calculate values
for index, row in df.iterrows():
    trigger = str(row['Trigger'])
    value_type = str(row['Type'])
    value = row['Value']

    # Check if the row contains a Trigger condition and extract the condition name
    if "Trigger" in trigger:
        for cond in conditions_corrected:
            if cond in trigger and "Start" in trigger:
                current_condition_corrected = cond
            elif cond in trigger and "End" in trigger:
                current_condition_corrected = None

    # If there's a current condition, check for 'HbO' or 'HbR' in the type and append the value
    if current_condition_corrected:
        if 'HbO' in value_type and 'SlidingOverallHbO' not in value_type:  # Exclude average values
            time_series_data_corrected[current_condition_corrected]['HbO'].append(float(value))
        elif 'HbR' in value_type and 'SlidingOverallHbR' not in value_type:  # Exclude average values
            time_series_data_corrected[current_condition_corrected]['HbR'].append(float(value))

# Handle uneven data by truncating longer series to match shorter ones
for condition, data in time_series_data_corrected.items():
    min_length = min(len(data['HbO']), len(data['HbR']))
    data['HbO'] = data['HbO'][:min_length]
    data['HbR'] = data['HbR'][:min_length]
    average_data_corrected[condition]['HbO'] = sum(data['HbO']) / len(data['HbO']) if data['HbO'] else 0.0
    average_data_corrected[condition]['HbR'] = sum(data['HbR']) / len(data['HbR']) if data['HbR'] else 0.0

# Line Plot for Time Series
for condition, data in time_series_data_corrected.items():
    plt.plot(data['HbO'], label=f'{condition} HbO')
    plt.plot(data['HbR'], label=f'{condition} HbR')

plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Average Change Over Time')
plt.legend()
plt.show()

# Bar Plot for Averages
for condition, data in average_data_corrected.items():
    plt.bar(f'{condition} HbO', data['HbO'])
    plt.bar(f'{condition} HbR', data['HbR'])

plt.xlabel('Condition')
plt.ylabel('Average Value')
plt.title('Contrast of Conditions')
plt.show()

# Histograms for Data Distribution
for condition, data in time_series_data_corrected.items():
    plt.hist(data['HbO'], alpha=0.5, label=f'{condition} HbO')
    plt.hist(data['HbR'], alpha=0.5, label=f'{condition} HbR')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f'Histogram for {condition}')
    plt.legend()
    plt.show()