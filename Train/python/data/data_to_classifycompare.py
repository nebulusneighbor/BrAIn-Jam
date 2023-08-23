import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load the CSV file, skipping the first 14 lines and only loading the first column
df = pd.read_csv('data.csv', skiprows=41, usecols=[0], header=None)

# Define the start and end cells for each condition
conditions = {
    'Control': ('Control Beginning', 'Control End', 3),
    'Condition 1': ('Trigger Condition 1 Start', 'Trigger Condition 1 End', 1),
    'Condition 2': ('Trigger Condition 2 Start', 'Trigger Condition 2 End', 2),
    'Condition 3': ('Trigger Condition 3 Start', 'Trigger Condition 3 End', 3),
}

data_chunks = []
labels = []

# Analyze each condition
for condition, (start, end, label) in conditions.items():
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

        # Split the data into overlapping chunks of 100 values
        for i in range(len(condition_data) - 99):
            chunk = condition_data[i:i+100]
            data_chunks.append(chunk.values)
            labels.append(label)

# Convert lists to numpy arrays
data_chunks = np.array(data_chunks)
labels = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_chunks, labels, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the SVM model
svm = SVC(kernel='linear')

# Train the SVM model
svm.fit(X_train, y_train)

# Predict the response for test dataset
y_pred_svm = svm.predict(X_test)

# Model Accuracy
print("SVM Accuracy:", metrics.accuracy_score(y_test, y_pred_svm))
print("SVM Precision:", metrics.precision_score(y_test, y_pred_svm, average='weighted'))
print("SVM Recall:", metrics.recall_score(y_test, y_pred_svm, average='weighted'))
print("SVM F1 Score:", metrics.f1_score(y_test, y_pred_svm, average='weighted'))

# Define the RF model
rf = RandomForestClassifier(n_estimators=100)

# Train the RF model
rf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred_rf = rf.predict(X_test)

# Model Accuracy
print("RF Accuracy:", metrics.accuracy_score(y_test, y_pred_rf))
print("RF Precision:", metrics.precision_score(y_test, y_pred_rf, average='weighted'))
print("RF Recall:", metrics.recall_score(y_test, y_pred_rf, average='weighted'))
print("RF F1 Score:", metrics.f1_score(y_test, y_pred_rf, average='weighted'))

# Define the NN model
nn = Sequential()
nn.add(Dense(64, input_dim=100, activation='relu'))
nn.add(Dense(64, activation='relu'))
nn.add(Dense(4, activation='softmax'))  # Assuming there are 4 classes (1, 2, and 3)

# Compile the NN model
nn.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train the NN model
nn.fit(X_train, y_train, epochs=10, batch_size=32)

# Predict the response for test dataset
y_pred_nn = np.argmax(nn.predict(X_test), axis=-1)

# Model Accuracy
print("NN Accuracy:", metrics.accuracy_score(y_test, y_pred_nn))
print("NN Precision:", metrics.precision_score(y_test, y_pred_nn, average='weighted'))
print("NN Recall:", metrics.recall_score(y_test, y_pred_nn, average='weighted'))
print("NN F1 Score:", metrics.f1_score(y_test, y_pred_nn, average='weighted'))
