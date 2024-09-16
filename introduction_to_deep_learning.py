# Introduction to Deep Learning

# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Brief overview of deep learning
print("Deep learning is a subset of machine learning that involves the use of artificial neural networks to model and solve complex problems.")

# Example of a simple neural network
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
