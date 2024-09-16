# Convolutional Neural Networks (CNNs)

# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

# Load the digits dataset
# The digits dataset is a collection of 1797 images of handwritten digits (0-9) in 8x8 pixel format.
# We reshape the images to (8, 8, 1) to prepare them for the CNN model.
digits = load_digits()
X = digits.images.reshape((digits.images.shape[0], 8, 8, 1))
y = digits.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, 64)).reshape(-1, 8, 8, 1)
X_test = scaler.transform(X_test.reshape(-1, 64)).reshape(-1, 8, 8, 1)

# Define a CNN model
# Define a CNN model
# This model consists of the following layers:
# 1. A convolutional layer with 32 filters, kernel size (3, 3), and ReLU activation.
# 2. A max pooling layer with pool size (2, 2) to downsample the feature maps.
# 3. A flatten layer to prepare the output for the dense layers.
# 4. A dense layer with 64 neurons and ReLU activation.
# 5. A final dense layer with 10 neurons and softmax activation for output probabilities.
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(8, 8, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
