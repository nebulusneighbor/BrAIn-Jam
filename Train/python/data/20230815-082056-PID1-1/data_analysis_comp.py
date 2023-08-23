import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('data_clean_2x.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Initialize variables
conditions = ['Control', 'Condition 1', 'Condition 2', 'Condition 3']
time_series_data = {condition: {'HbO': [], 'HbR': []} for condition in conditions}
average_data = {condition: {'HbO': 0.0, 'HbR': 0.0} for condition in conditions}
current_condition = None

# Iterate through rows to identify conditions and calculate values
for index, row in df.iterrows():
    trigger = str(row['Trigger'])
    value_type = str(row['Type'])
    value = row['Value']
    
    if "Trigger" in trigger:
        for condition in conditions:
            if condition in trigger and "Start" in trigger:
                current_condition = condition
            elif condition in trigger and "End" in trigger:
                current_condition = None
    
    if current_condition:
        if 'HbO' in value_type:
            time_series_data[current_condition]['HbO'].append(float(value))
        elif 'HbR' in value_type:
            time_series_data[current_condition]['HbR'].append(float(value))

# Calculate averages and prepare time series
for condition, data in time_series_data.items():
    average_data[condition]['HbO'] = sum(data['HbO']) / len(data['HbO']) if data['HbO'] else 0.0
    average_data[condition]['HbR'] = sum(data['HbR']) / len(data['HbR']) if data['HbR'] else 0.0

# Line Plot for Time Series
for condition, data in time_series_data.items():
    plt.plot(data['HbO'], label=f'{condition} HbO')
    plt.plot(data['HbR'], label=f'{condition} HbR')

plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Average Change Over Time')
plt.legend()
plt.show()

# Bar Plot for Averages
for condition, data in average_data.items():
    plt.bar(f'{condition} HbO', data['HbO'])
    plt.bar(f'{condition} HbR', data['HbR'])

plt.xlabel('Condition')
plt.ylabel('Average Value')
plt.title('Contrast of Conditions')
plt.show()

# Histograms for Data Distribution
for condition, data in time_series_data.items():
    plt.hist(data['HbO'], alpha=0.5, label=f'{condition} HbO')
    plt.hist(data['HbR'], alpha=0.5, label=f'{condition} HbR')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f'Histogram for {condition}')
    plt.legend()
    plt.show()
