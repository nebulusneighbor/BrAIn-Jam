import pandas as pd
import matplotlib.pyplot as plt
import csv

rows = []
with open('Sliding_Avg.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        # Extract the timestamp and trigger directly from the appropriate columns
        timestamp = row[0].strip()
        trigger = row[1].strip()

        if not timestamp or not trigger:
            print("Problematic row:", row)
            continue

        # Append the row in the desired format
        rows.append([timestamp, trigger] + row[2:])

# Convert the rows to a DataFrame, skipping the header row
df = pd.DataFrame(rows[1:], columns=['Timestamp', 'Trigger', 'Type Mean', 'Value.1'])

# The rest of the code can remain the same



# Define a function to get the data between the specified start and end triggers
def extract_condition_data(df, start_trigger, end_trigger):
    start_rows = df[df['Trigger'].str.contains(start_trigger, regex=False)]
    end_rows = df[df['Trigger'].str.contains(end_trigger, regex=False)]
    
    if start_rows.empty or end_rows.empty:
        print(f"No rows found for start trigger: {start_trigger} and end trigger: {end_trigger}")
        return pd.DataFrame() # Return an empty DataFrame

    start_index = start_rows.index[0]
    end_index = end_rows.index[-1]

    return df.loc[start_index+1:end_index-1]

# Separate data by HbO and HbR
df_HbO = df[df['Type Mean'] == 'SlidingOverallHbO']
df_HbR = df[df['Type Mean'] == 'SlidingOverallHbR']

# Define the triggers for each condition
conditions = {
    'Control': ("Trigger Control Condition Start", "Trigger Control Condition End"),
    'Condition 1': ("Trigger Condition 1 Start", "Trigger Condition 1 End"),
    'Condition 2': ("Trigger Condition 2 Start", "Trigger Condition 2 End"),
    'Condition 3': ("Trigger Condition 3 Start", "Trigger Condition 3 End"),
}


# Iterate through conditions to plot graphs
# Iterate through conditions to plot graphs
for condition_name, triggers in conditions.items():
    start_trigger, end_trigger = triggers
    condition_HbO = extract_condition_data(df_HbO, start_trigger, end_trigger)
    condition_HbR = extract_condition_data(df_HbR, start_trigger, end_trigger)

    if condition_HbO.empty or condition_HbR.empty:
        print(f"Skipping {condition_name} as no data was found.")
        continue

    # Compute averages
    mean_HbO = condition_HbO['Value.1'].mean()
    mean_HbR = condition_HbR['Value.1'].mean()

    # Plot averages for the condition
    plt.figure()
    plt.bar(['HbO', 'HbR'], [mean_HbO, mean_HbR], color=['red', 'blue'])
    plt.title(f'{condition_name} Averages')
    plt.ylabel('Value')
    plt.show()

    # Plot time-series for the condition
    plt.figure()
    plt.plot(pd.to_datetime(condition_HbO['Timestamp']), condition_HbO['Value.1'], color='red', label='HbO')
    plt.plot(pd.to_datetime(condition_HbR['Timestamp']), condition_HbR['Value.1'], color='blue', label='HbR')
    plt.title(f'{condition_name} Over Time')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
