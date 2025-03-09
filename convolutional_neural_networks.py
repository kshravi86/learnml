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

"""
Mathematical Background of the CNN:

1. Convolutional Layer:
   - Input: 8x8x1 image
   - Convolution operation: f(i,j) = Σ Σ (kernel(m,n) * input(i-m,j-n))
   - Output size = ((W-K+2P)/S)+1 where:
     W = input size
     K = kernel size
     P = padding
     S = stride

2. Max Pooling:
   - Reduces spatial dimensions by taking maximum value in each window
   - Output size = input_size/pool_size

3. Dense Layer:
   - Applies weight matrix W and bias b: y = activation(Wx + b)
   - ReLU activation: f(x) = max(0,x)
   - Softmax activation: σ(x)i = exp(xi) / Σ exp(xj)

4. Loss Function:
   - Sparse Categorical Cross Entropy: -Σ y_true * log(y_pred)
"""

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Print detailed mathematical computations
def print_model_mathematics():
    print("\nDetailed Mathematical Computations:")
    print("===================================")
    
    # Convolutional Layer calculations
    input_size = (8, 8, 1)
    kernel_size = (3, 3)
    conv_stride = 1
    conv_output_size = ((input_size[0] - kernel_size[0]) // conv_stride + 1,
                       (input_size[1] - kernel_size[1]) // conv_stride + 1,
                       32)
    
    print(f"1. Convolutional Layer:")
    print(f"   Input size: {input_size}")
    print(f"   Output size: {conv_output_size}")
    print(f"   Parameters: {32 * 3 * 3 * 1 + 32} (weights + biases)")
    
    # MaxPooling calculations
    pool_size = (2, 2)
    pool_output_size = (conv_output_size[0] // 2,
                       conv_output_size[1] // 2,
                       conv_output_size[2])
    
    print(f"\n2. MaxPooling Layer:")
    print(f"   Input size: {conv_output_size}")
    print(f"   Output size: {pool_output_size}")
    
    # Dense layer calculations
    flattened_size = pool_output_size[0] * pool_output_size[1] * pool_output_size[2]
    dense1_params = flattened_size * 64 + 64
    dense2_params = 64 * 10 + 10
    
    print(f"\n3. Dense Layers:")
    print(f"   First Dense Layer:")
    print(f"   - Input size: {flattened_size}")
    print(f"   - Output size: 64")
    print(f"   - Parameters: {dense1_params}")
    
    print(f"\n   Second Dense Layer (Output):")
    print(f"   - Input size: 64")
    print(f"   - Output size: 10")
    print(f"   - Parameters: {dense2_params}")
    
    total_params = (32 * 3 * 3 * 1 + 32) + dense1_params + dense2_params
    print(f"\nTotal trainable parameters: {total_params}")

print_model_mathematics()
