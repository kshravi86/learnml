import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases with random values
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.01
        self.bias1 = np.zeros((1, hidden_size))
        self.bias2 = np.zeros((1, output_size))
        
        # Storage for intermediate values (needed for backpropagation)
        self.z1 = None  # Input to hidden layer before activation
        self.a1 = None  # Hidden layer activation
        self.z2 = None  # Hidden to output layer before activation
        self.a2 = None  # Output layer activation

    def sigmoid(self, x):
        """
        Sigmoid activation function
        f(x) = 1 / (1 + e^(-x))
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """
        Derivative of sigmoid function
        f'(x) = f(x) * (1 - f(x))
        """
        return x * (1 - x)

    def forward(self, X):
        """
        Forward propagation
        X: input data
        """
        # First layer computation
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = self.sigmoid(self.z1)
        
        # Second layer computation
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2

    def backward(self, X, y, output, learning_rate):
        """
        Backward propagation
        X: input data
        y: true labels
        output: predicted output
        learning_rate: learning rate for gradient descent
        """
        m = X.shape[0]  # Number of training examples
        
        # Calculate gradients for second layer
        delta2 = (output - y) * self.sigmoid_derivative(output)
        dW2 = np.dot(self.a1.T, delta2)
        db2 = np.sum(delta2, axis=0, keepdims=True)
        
        # Calculate gradients for first layer
        delta1 = np.dot(delta2, self.weights2.T) * self.sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, delta1)
        db1 = np.sum(delta1, axis=0, keepdims=True)
        
        # Update weights and biases
        self.weights2 -= learning_rate * dW2
        self.bias2 -= learning_rate * db2
        self.weights1 -= learning_rate * dW1
        self.bias1 -= learning_rate * db1

    def train(self, X, y, epochs, learning_rate):
        """
        Train the neural network
        X: input data
        y: true labels
        epochs: number of training iterations
        learning_rate: learning rate for gradient descent
        """
        for epoch in range(epochs):
            # Forward propagation
            output = self.forward(X)
            
            # Backward propagation
            self.backward(X, y, output, learning_rate)
            
            # Calculate and print loss every 100 epochs
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss: {loss}")

# Example usage
if __name__ == "__main__":
    # Generate some example data (XOR problem)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Create and train the neural network
    nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    nn.train(X, y, epochs=1000, learning_rate=0.1)

    # Test the network
    print("\nTesting the neural network:")
    print("Input (0,0) -> Output:", nn.forward(np.array([[0, 0]])))
    print("Input (0,1) -> Output:", nn.forward(np.array([[0, 1]])))
    print("Input (1,0) -> Output:", nn.forward(np.array([[1, 0]])))
    print("Input (1,1) -> Output:", nn.forward(np.array([[1, 1]])))