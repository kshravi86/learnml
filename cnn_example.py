# Import the NumPy library, which provides support for large, multi-dimensional arrays and matrices for numerical computations.
# This library is essential for efficient numerical computations in machine learning.
import numpy as np
# Import the necessary modules from the Keras library, which provides a high-level neural networks API.
# Import the necessary modules from the Keras library, which provides a high-level neural networks API.
# These modules are used to define the Convolutional Neural Network (CNN) model architecture.
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define the Convolutional Neural Network (CNN) model using the Sequential API.
# The model consists of multiple convolutional and max pooling layers, followed by fully connected layers.
model = Sequential()
# Add a convolutional layer with 32 filters, kernel size 3x3, and ReLU activation.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# Add a max pooling layer with pool size 2x2.
model.add(MaxPooling2D((2, 2)))
# Add another convolutional layer with 64 filters, kernel size 3x3, and ReLU activation.
model.add(Conv2D(64, (3, 3), activation='relu'))
# Add another max pooling layer with pool size 2x2.
model.add(MaxPooling2D((2, 2)))
# Add a third convolutional layer with 64 filters, kernel size 3x3, and ReLU activation.
model.add(Conv2D(64, (3, 3), activation='relu'))
# Flatten the output of the convolutional layers to prepare for fully connected layers.
model.add(Flatten())
# Add a fully connected layer with 64 units and ReLU activation.
model.add(Dense(64, activation='relu'))
# Add a final fully connected layer with 10 units and softmax activation for output.
model.add(Dense(10, activation='softmax'))
# Import keras
# Import the Keras library, which provides a high-level neural networks API.
import keras

# Load the dataset (e.g. CIFAR-10)
# Load the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes.
from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
# Normalize the pixel values of the training and testing datasets to be between 0 and 1.
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Convert class vectors to binary class matrices
# Convert the class vectors to binary class matrices using the to_categorical function.
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Compile the model
# Compile the model with the Adam optimizer and categorical cross-entropy loss function.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load the dataset (e.g. CIFAR-10)
from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# Normalize pixel values to be between 0 and 1
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Train the model
# Train the model on the training dataset with 10 epochs, batch size 32, and validation on the testing dataset.
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=2)

# Evaluate the model
# Evaluate the model on the testing dataset and print the test accuracy.
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy:.2f}')
