# Introduction to Deep Learning

# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Brief overview of deep learning
print("Deep learning is a subset of machine learning that involves the use of artificial neural networks to model and solve complex problems.")
# It is particularly useful for tasks such as image and speech recognition, natural language processing, and game playing.

# Example of a simple neural network
# Example of a simple neural network
# This model consists of three fully connected (dense) layers:
# 1. The first layer has 64 neurons, uses the ReLU activation function, and takes input of shape (784,).
# 2. The second layer has 32 neurons, uses the ReLU activation function.
# 3. The third layer has 10 neurons, uses the softmax activation function for output probabilities.
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
