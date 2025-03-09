# Introduction to Deep Learning

# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Brief overview of deep learning
print("Deep learning is a subset of machine learning that involves the use of artificial neural networks to model and solve complex problems.")
# It is particularly useful for tasks such as image and speech recognition, natural language processing, and game playing.

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

# Mathematical Explanation of the Neural Network
print("\nMathematical Components of the Neural Network:")
print("1. Dense Layer Mathematics:")
print("   For each neuron: output = activation_function(Σ(weights * inputs) + bias)")
print("   - Input shape: 784 dimensions (x₁, x₂, ..., x₇₈₄)")
print("   - Weight matrix dimensions: (784, 64) for first layer")

print("\n2. ReLU Activation Function:")
print("   f(x) = max(0, x)")
print("   - Mathematically: f(x) = { x if x > 0")
print("                             { 0 if x ≤ 0")

print("\n3. Softmax Activation (Final Layer):")
print("   softmax(x)ᵢ = exp(xᵢ) / Σⱼ exp(xⱼ)")
print("   - Converts outputs to probabilities that sum to 1")

print("\n4. Loss Function (Sparse Categorical Cross-Entropy):")
print("   L = -Σᵢ yᵢ log(ŷᵢ)")
print("   where yᵢ is true label and ŷᵢ is predicted probability")

print("\n5. Adam Optimizer:")
print("   - Uses adaptive learning rates")
print("   - Updates weights: w = w - η * m̂ₜ / (√v̂ₜ + ε)")
print("   where η is learning rate, m̂ₜ is bias-corrected first moment")
print("   and v̂ₜ is bias-corrected second moment")

print("\n6. Gradient Descent Visualization:")
print("     Loss")
print("      ↑     ·")
print("      |   · · ·")
print("      | ·     · ·")
print("      |·         · ·")
print("      |             · →  Minimum")
print("      |----------------------→ Weights")

print("\n7. Neural Network Layer Structure:")
print("    [Input]     [Hidden]    [Output]")
print("      O --------→ O --------→ O")
print("      O --------→ O --------→ O")
print("      O --------→ O --------→ O")
print("    784 nodes   64 nodes   10 nodes")

print("\n8. Backpropagation Mathematics:")
print("   ∂E/∂w = ∂E/∂y * ∂y/∂z * ∂z/∂w")
print("   where: E = error, w = weights")
print("          y = activation output")
print("          z = weighted sum")

print("\n9. Learning Rate Effect:")
print("   Small η: Slow but stable convergence")
print("   Large η: Fast but might overshoot")
print("   η = learning rate in: w = w - η * ∇w")

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model architecture summary
model.summary()
