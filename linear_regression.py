# This code implements a simple linear regression model from scratch using Python and NumPy. 
# The goal is to create a model that learns a linear relationship between input data (features) and output data (targets) 
# by minimizing the error using gradient descent. The model is then used to predict values for given input data.

import numpy as np

# Define the linear regression model
class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        # Initialize the learning rate and the number of iterations
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None  # Weights for the features (initialized later)
        self.bias = None  # Bias term (initialized later)

    def fit(self, X, y):
        # Get the number of samples (rows) and features (columns) from the input data
        n_samples, n_features = X.shape

        # Initialize parameters (weights and bias) to zero
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Perform gradient descent for a specified number of iterations
        for _ in range(self.n_iters):
            # Calculate the predicted values using the current weights and bias
            y_predicted = np.dot(X, self.weights) + self.bias

            # Compute the gradient for weights
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            # Compute the gradient for bias
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update the weights and bias using the gradients and learning rate
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        # Calculate the predicted values using the learned weights and bias
        y_approximated = np.dot(X, self.weights) + self.bias
        return y_approximated

# Example usage
X = np.array([[1], [2], [3], [4], [5]])  # Feature matrix (input data)
y = np.array([2, 4, 5, 4, 5])  # Target values (output data)

# Create an instance of the LinearRegression model
model = LinearRegression()
# Fit the model to the data (train the model)
model.fit(X, y)
# Predict the values for the input data
predicted = model.predict(X)
# Print the predicted values
print(predicted)
