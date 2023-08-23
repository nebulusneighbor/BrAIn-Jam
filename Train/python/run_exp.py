from pythonosc import dispatcher, osc_server
from pylsl import StreamInfo, StreamOutlet, local_clock
import keyboard
import csv
import time
import os
import random
from datetime import datetime
import tkinter as tk
import fnmatch
import threading

#File notes: logs double markers a beginning marker gets put at the top of the csv

# Create a new StreamInfo for the stream
info = StreamInfo('NIRS_Triggers', 'Triggers', 1, 0, 'int32', 'NSP2_19390109_A')

# Create a new StreamOutlet
outlet = StreamOutlet(info)

# Array to store incoming numbers from OSC messages
data = []
start_time = None
csv_data = []
timestamps = []
data_by_timestamp = {}

# Define regions and channels for HbO
regions = {
    'Frontopolar': ['HbO1', 'HbO2', 'HbO26', 'HbO27', 'HbO28', 'HbO32', 'HbO34'],
    'DLPFC_Left': ['HbO3', 'HbO4', 'HbO5', 'HbO7', 'HbO8', 'HbO14', 'HbO24'],
    'DLPFC_Right': ['HbO30', 'HbO33', 'HbO35', 'HbO36', 'HbO37', 'HbO39', 'HbO42'],
    'MPFC': ['HbO10', 'HbO11', 'HbO12', 'HbO13', 'HbO41'],
    'TPJ_Left': ['HbO16', 'HbO17', 'HbO18', 'HbO19'],
    'TPJ_Right': ['HbO46', 'HbO47', 'HbO49', 'HbO50']
}
# Add HbR channels to regions
for region, channels in regions.items():
    regions[region].extend([channel.replace('HbO', 'HbR') for channel in channels])

# HbO channel mapping based on the provided information
hbo_channel_mapping = [
    1, 2, 26, 27, 28, 32, 34, # Frontopolar
    3, 4, 5, 7, 8, 14, 24, 30, 33, 35, 36, 37, 39, 42, # DLPFC
    10, 11, 12, 13, 41, # MPFC
    16, 17, 18, 19, 46, 47, 49, 50 # TPJ
]

# HbR channel mapping (HbR channels are 50 more than corresponding HbO channels)
hbr_channel_mapping = [channel + 50 for channel in hbo_channel_mapping]

# Data collection by region
data_by_region = {region: {channel: [] for channel in channels} for region, channels in regions.items()}


# Define a function to write data to CSV
def write_to_csv():
    with open(csv_file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        for timestamp, data in list(data_by_timestamp.items()): # Iterate through a copy of the items
            trigger = data.get('Trigger', '') # Extracting the trigger tuple
            if trigger: # If trigger is present, write it first
                writer.writerow([timestamp, str(trigger), '', '', '', ''])

            # Write raw channel data
            for region, channels in regions.items():
                for channel in channels:
                    value = data.get(f'{region}_{channel}', '')
                    if value:  # Check if the value is present
                        channel_type = channel.split('_')[0] # Extracting HbO or HbR
                        row = [timestamp, str(trigger) if trigger else '', f'{region}_{channel_type}', value, '', '', '', ''] # Using trigger tuple
                        writer.writerow(row)

            # Write sliding overall means
            for mean_type in ['SlidingOverallHbO', 'SlidingOverallHbR']:
                slide_mean_value = data.get(mean_type, '')
                if slide_mean_value:  # Check if the value is present
                    row = [timestamp, str(trigger) if trigger else '', '', '', mean_type, slide_mean_value, '', ''] # Using trigger tuple
                    writer.writerow(row)

            # Remove the processed timestamp from the dictionary
            del data_by_timestamp[timestamp]

# Creating CSV file at the beginning
folder_path = os.path.join("C:\\Users\\ptgyo\\OneDrive\\Documents\\JamBrain\\Data_pythonCapture", datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(folder_path, exist_ok=True)
csv_file_path = os.path.join(folder_path, 'data.csv')

# Open the CSV file and write the header
with open(csv_file_path, 'w', newline='') as f:
    writer = csv.writer(f)
    headers = ['Timestamp', 'Trigger', 'Type', 'Value', 'Type Mean', 'Value']
    writer.writerow(headers)

# Change timestamp resolution to include milliseconds
def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def handle_data(timestamp, key, value):
    if timestamp not in data_by_timestamp:
        data_by_timestamp[timestamp] = {}
    elif key.endswith('Mean'):
        region, mean_type = key.split('_')
        data_by_timestamp[timestamp][f'{region}_{mean_type}_Mean'] = value
    data_by_timestamp[timestamp][key] = value

def handle_hbo_raw(address, *values):
    timestamp = get_timestamp()
    timestamps.append(timestamp)
    channel_idx = int(address.split('/')[-1].replace('Channel', ''))
    actual_channel_number = hbo_channel_mapping[channel_idx - 1]
    channel_name = f'HbO{actual_channel_number}'
    for region, channels in regions.items():
        if channel_name in channels:
            handle_data(timestamp, f'{region}_{channel_name}', values[0])
            break

def handle_hbr_raw(address, *values):
    timestamp = get_timestamp()
    channel_idx = int(address.split('/')[-1].replace('Channel', ''))
    actual_channel_number = hbr_channel_mapping[channel_idx - 1]
    channel_name = f'HbR{actual_channel_number - 50}'
    for region, channels in regions.items():
        if channel_name in channels:
            handle_data(timestamp, f'{region}_{channel_name}', values[0])
            break

def handle_hbo_mean(address, *values):
    timestamp = get_timestamp()
    region = address.split('/')[-1]
    mean_value = values[0]
    handle_data(timestamp, f'{region}_HbO_Mean', mean_value)

def handle_hbr_mean(address, *values):
    timestamp = get_timestamp()
    region = address.split('/')[-1]
    mean_value = values[0]
    handle_data(timestamp, f'{region}_HbR_Mean', mean_value)

def handle_sliding_overall_hbo(address, *values):
    timestamp = get_timestamp()
    sliding_overall_mean_hbo = values[0]
    handle_data(timestamp, 'SlidingOverallHbO', sliding_overall_mean_hbo)

def handle_sliding_overall_hbr(address, *values):
    timestamp = get_timestamp()
    sliding_overall_mean_hbr = values[0]
    handle_data(timestamp, 'SlidingOverallHbR', sliding_overall_mean_hbr)

def print_osc_message(address, *args):
    print(f"Received message from {address}: {args}")
    for arg in args:
        if fnmatch.fnmatch(address, "/RawReading/HbO/Channel*"):
            handle_hbo_raw(address, arg)
            #print('received message O')
        if fnmatch.fnmatch(address, "/RawReading/HbR/Channel*"):
            handle_hbr_raw(address, arg)
            #print('received message R')

def send_lsl_marker(marker, marker_name):
    global start_time
    timestamp = get_timestamp()
    print(f"Sending LSL marker: {marker}")
    outlet.push_sample([marker], local_clock())
    if marker == 1:
        start_time = local_clock()
    # Initialize the key if not present
    if timestamp not in data_by_timestamp:
        data_by_timestamp[timestamp] = {}

    # Check if the marker_name is for the control condition
    if 'Control' in marker_name:
        trigger_name = f'Trigger {marker_name}' # Naming for the control condition
    else:
        trigger_name = marker_name

    # Now you can safely assign the value
    data_by_timestamp[timestamp]['Trigger'] = (marker_name, timestamp)
    print(f"Assigning trigger: {marker_name} at timestamp: {timestamp}")

def write_beginning_marker(writer, marker_name, timestamp):
    writer.writerow([timestamp, str((marker_name, timestamp)), '', '', '', ''])

def update_gui(window, label_current, label_next, label_timer, condition, next_condition, seconds):
    label_current.config(text=f"Current: {'Rest' if condition == 0 else f'Condition {condition}'}")
    label_next.config(text=f"Next: {'Rest' if next_condition == 0 else f'Condition {next_condition}'}")
    label_timer.config(text=f"Time Remaining: {seconds} seconds")
    window.update()

def experiment(window, label_current, label_next, label_timer):
    try:
        conditions = ['Condition 1', 'Condition 2', 'Condition 3']
        control_condition = 'Control Condition'
        current_state = 'Rest'
        next_state = control_condition # Start with control condition
        condition_counter = 0
        rest_interval = False
        while True:
            if current_state == 'Rest':
                for i in range(10, 0, -1): # Rest time 10s
                    update_gui(window, label_current, label_next, label_timer, 0, next_state, i)
                    time.sleep(1)
                current_state = next_state
                next_state = 'Rest'
            else:
                if current_state == control_condition:
                    duration = 60 #60
                    start_marker = 7
                    end_marker = 8
                else:
                    condition_number = int(current_state.split()[-1])
                    duration = 120 #120
                    start_marker = 2 + (condition_number - 1) * 2
                    end_marker = 3 + (condition_number - 1) * 2

                send_lsl_marker(start_marker, f'Trigger {current_state} Start')
                for i in range(duration, 0, -1):
                    update_gui(window, label_current, label_next, label_timer, current_state, next_state, i)
                    time.sleep(1)
                send_lsl_marker(end_marker, f'Trigger {current_state} End')

                if condition_counter % 10 == 0 and condition_counter != 0:
                    write_to_csv()
                    next_state = control_condition
                else:
                    next_state = random.choice(conditions)

                current_state = 'Rest'
                condition_counter += 1
    except Exception as e:
        print(f"An exception occurred: {e}")

def start_experiment(window, label_current, label_next, label_timer):
    timestamp = get_timestamp() # Get the current timestamp
    send_lsl_marker(1, 'Beginning')
    with open(os.path.join(folder_path, 'data.csv'), 'a', newline='') as f: # Open the CSV file in append mode
        writer = csv.writer(f)
        write_beginning_marker(writer, 'Beginning', timestamp) # Write the beginning marker
    experiment_thread = threading.Thread(target=experiment, args=(window, label_current, label_next, label_timer)) # Create a new thread for the experiment
    experiment_thread.daemon = True # Set the thread as a daemon, so it will close when the main program closes
    experiment_thread.start() # Start the experiment in a new thread

def create_gui():
    window = tk.Tk()
    window.title("Experiment Timer")
    label_current = tk.Label(window, text="Current: ")
    label_current.pack()
    label_next = tk.Label(window, text="Next: ")
    label_next.pack()
    label_timer = tk.Label(window, text="Time Remaining: ")
    label_timer.pack()
    button_start = tk.Button(window, text="Start Experiment", command=lambda: start_experiment(window, label_current, label_next, label_timer))
    button_start.pack()
    window.mainloop()

dispatcher = dispatcher.Dispatcher()
dispatcher.map("/Density/HbO/*", handle_hbo_raw)
dispatcher.map("/Density/HbR/*", handle_hbr_raw)
dispatcher.map("/HbO_Mean/*", handle_hbo_mean)
dispatcher.map("/HbR_Mean/*", handle_hbr_mean)
dispatcher.map("/SlidingOverallHbOMean", handle_sliding_overall_hbo)
dispatcher.map("/SlidingOverallHbRMean", handle_sliding_overall_hbr)
dispatcher.map("/*", print_osc_message)

server = osc_server.ThreadingOSCUDPServer(("localhost", 8888), dispatcher)
print("Serving on {}".format(server.server_address))

keyboard.on_press_key("0", lambda _: create_gui())

# Saving data including HBO and HBR channels
try:
    server.serve_forever()
except KeyboardInterrupt:
    print("Server stopped.")
    write_to_csv()
