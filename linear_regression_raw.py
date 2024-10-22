import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Initialize Linear Regression model
        
        Parameters:
            learning_rate (float): How much to update parameters in each step
            n_iterations (int): Number of training iterations
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.history = {'loss': []}  # For tracking training progress
    
    def _mean_squared_error(self, y_true, y_predicted):
        """Calculate the mean squared error between true and predicted values"""
        return np.mean((y_true - y_predicted) ** 2)
    
    def fit(self, X, y):
        """
        Train the model using gradient descent
        
        Parameters:
            X (array-like): Training data of shape (n_samples, n_features)
            y (array-like): Target values of shape (n_samples,)
        """
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for iteration in range(self.n_iterations):
            # Forward pass (make predictions)
            y_predicted = self._predict(X)
            
            # Calculate gradients
            # dw = ∂(MSE)/∂w = (2/n) * X^T * (y_predicted - y)
            # db = ∂(MSE)/∂b = (2/n) * sum(y_predicted - y)
            dw = (2/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (2/n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Track loss
            loss = self._mean_squared_error(y, y_predicted)
            self.history['loss'].append(loss)
            
            # Optional: Print progress every 100 iterations
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Loss = {loss:.4f}")
    
    def _predict(self, X):
        """Make predictions using current parameters"""
        return np.dot(X, self.weights) + self.bias
    
    def predict(self, X):
        """
        Make predictions for new data
        
        Parameters:
            X (array-like): Data to make predictions for, shape (n_samples, n_features)
        
        Returns:
            array-like: Predicted values
        """
        return self._predict(X)
    
    def score(self, X, y):
        """
        Calculate R² score (coefficient of determination)
        
        Parameters:
            X (array-like): Test data
            y (array-like): True values
        
        Returns:
            float: R² score
        """
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(0)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1) * 0.1
    
    # Create and train model
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y.flatten())
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Print results
    print("\nTraining complete!")
    print(f"Final weights: {model.weights}")
    print(f"Final bias: {model.bias}")
    print(f"R² score: {model.score(X, y.flatten()):.4f}")
