# Recurrent Neural Networks (RNNs)

# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate a sample sequence dataset
import numpy as np
np.random.seed(42)
timesteps = 10
features = 10
samples = 1000
X = np.random.rand(samples, timesteps, features)
y = np.random.rand(samples, timesteps, features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, features)).reshape(-1, timesteps, features)
X_test = scaler.transform(X_test.reshape(-1, features)).reshape(-1, timesteps, features)

# Define an RNN model
model = keras.Sequential([
    keras.layers.SimpleRNN(64, input_shape=(timesteps, features)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(features)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
