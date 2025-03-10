# Recurrent Neural Networks (RNNs)

# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate a sample sequence dataset
# We create a random dataset of sequences with 10 timesteps, 10 features, and 1000 samples.
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
# Define an RNN model
# This model consists of the following layers:
# 1. A simple RNN layer with 64 units, taking input of shape (timesteps, features).
# 2. A dense layer with 64 neurons and ReLU activation.
# 3. A final dense layer with features neurons for output.
model = keras.Sequential([
    keras.layers.SimpleRNN(64, input_shape=(timesteps, features)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(features)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Mathematical Explanation of RNN:
# -----------------------------
# For each time step t, RNN computes:
# h[t] = tanh(W_hh * h[t-1] + W_xh * x[t] + b_h)  # hidden state
# y[t] = W_hy * h[t] + b_y                         # output
#
# Where:
# - h[t]: hidden state at time t
# - x[t]: input at time t
# - y[t]: output at time t
# - W_hh: weight matrix for hidden-to-hidden connections
# - W_xh: weight matrix for input-to-hidden connections
# - W_hy: weight matrix for hidden-to-output connections
# - b_h: hidden bias vector
# - b_y: output bias vector
# - tanh: hyperbolic tangent activation function
#
# Loss Function (Mean Squared Error):
# MSE = (1/n) * Σ(y_true - y_pred)²
#
# Backpropagation Through Time (BPTT):
# The gradient at each time step t is:
# ∂E/∂W = Σ(∂E/∂y[t] * ∂y[t]/∂h[t] * ∂h[t]/∂W)
# where E is the error/loss function

# Print mathematical explanations
print("RNN Mathematical Formulas:")
print("-------------------------")
print("For each time step t, RNN computes:")
print("h[t] = tanh(W_hh * h[t-1] + W_xh * x[t] + b_h)  # hidden state")
print("y[t] = W_hy * h[t] + b_y                         # output")
print("\nWhere:")
print("- h[t]: hidden state at time t")
print("- x[t]: input at time t")
print("- y[t]: output at time t")
print("- W_hh: weight matrix for hidden-to-hidden connections")
print("- W_xh: weight matrix for input-to-hidden connections")
print("- W_hy: weight matrix for hidden-to-output connections")
print("- b_h: hidden bias vector")
print("- b_y: output bias vector")
print("- tanh: hyperbolic tangent activation function")
print("\nLoss Function (Mean Squared Error):")
print("MSE = (1/n) * Σ(y_true - y_pred)²")
print("\nBackpropagation Through Time (BPTT):")
print("The gradient at each time step t is:")
print("∂E/∂W = Σ(∂E/∂y[t] * ∂y[t]/∂h[t] * ∂h[t]/∂W)")
print("where E is the error/loss function")
