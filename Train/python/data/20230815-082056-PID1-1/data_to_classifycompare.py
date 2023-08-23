import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import time

# Start the timer
start_time = time.time()

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

# Create even chunks with overlapping data
chunk_size = 100  # Adjust the chunk size as needed
overlap = 50  # Adjust the overlap size as needed

data_chunks = []
labels = []

# Analyze each condition
for condition, data in time_series_data_corrected.items():
    hbo_values = data['HbO']
    hbr_values = data['HbR']

    # Calculate the number of chunks with the given chunk size and overlap
    num_chunks = (len(hbo_values) - chunk_size) // overlap + 1

    # Create even chunks with overlap
    for i in range(num_chunks):
        start_index = i * overlap
        end_index = start_index + chunk_size
        chunk = np.concatenate((hbo_values[start_index:end_index], hbr_values[start_index:end_index]))
        data_chunks.append(chunk)
        labels.append(condition)

# Convert lists to numpy arrays
data_chunks = np.array(data_chunks)
labels = np.array(labels)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Stop the timer after data preprocessing
preprocessing_time = time.time() - start_time
print("Data Preprocessing Time:", preprocessing_time, "seconds")

# Machine Learning Code (SVM)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_chunks, labels_encoded, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the SVM model
svm = SVC(kernel='linear')

# Train the SVM model
start_time_svm = time.time()
svm.fit(X_train, y_train)
svm_time = time.time() - start_time_svm

# Predict the response for test dataset
y_pred_svm = svm.predict(X_test)

# Model Accuracy
print("SVM Accuracy:", metrics.accuracy_score(y_test, y_pred_svm))
print("SVM Time:", svm_time, "seconds")

# Machine Learning Code (RF)

# Define the RF model
rf = RandomForestClassifier(n_estimators=100)

# Train the RF model
start_time_rf = time.time()
rf.fit(X_train, y_train)
rf_time = time.time() - start_time_rf

# Predict the response for test dataset
y_pred_rf = rf.predict(X_test)

# Model Accuracy
print("RF Accuracy:", metrics.accuracy_score(y_test, y_pred_rf))
print("RF Precision:", metrics.precision_score(y_test, y_pred_rf, average='weighted'))
print("RF Recall:", metrics.recall_score(y_test, y_pred_rf, average='weighted'))
print("RF F1 Score:", metrics.f1_score(y_test, y_pred_rf, average='weighted'))
print("RF Time:", rf_time, "seconds")

# Machine Learning Code (NN)

# Define the NN model
nn = Sequential()
nn.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
nn.add(Dense(64, activation='relu'))
nn.add(Dense(len(conditions_corrected), activation='softmax'))  # Output classes

# Compile the NN model
nn.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train the NN model
start_time_nn = time.time()
nn.fit(X_train, y_train, epochs=500, batch_size=32)
nn_time = time.time() - start_time_nn

# Predict the response for test dataset
y_pred_nn = np.argmax(nn.predict(X_test), axis=-1)

# Model Accuracy
print("NN Accuracy:", metrics.accuracy_score(y_test, y_pred_nn))
print("NN Precision:", metrics.precision_score(y_test, y_pred_nn, average='weighted'))
print("NN Recall:", metrics.recall_score(y_test, y_pred_nn, average='weighted'))
print("NN F1 Score:", metrics.f1_score(y_test, y_pred_nn, average='weighted'))
print("NN Time:", nn_time, "seconds")

# Stop the timer after machine learning
ml_time = time.time() - start_time - preprocessing_time


print("SVM Time:", svm_time, "seconds")
print("RF Time:", rf_time, "seconds")
print("NN Time:", nn_time, "seconds")
print("Machine Learning Time (Total):", ml_time, "seconds")

# Total elapsed time
total_time = time.time() - start_time
print("Total Elapsed Time:", total_time, "seconds")