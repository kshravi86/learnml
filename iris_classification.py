import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Log dataset information
logger.info("Loaded Iris dataset with %d samples and %d features", len(X), len(X.columns))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Log dataset split information
logger.info("Split dataset into training set with %d samples and testing set with %d samples", len(X_train), len(X_test))

# Scale the data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Log scaling information
logger.info("Scaled data using StandardScaler")

# Create a Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Log model creation
logger.info("Created Logistic Regression model with max_iter=%d", model.max_iter)

# Train the model
model.fit(X_train_scaled, y_train)

# Log training information
logger.info("Trained model on training set")

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Log prediction information
logger.info("Made predictions on testing set")

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
logger.info("Model accuracy: %.3f", accuracy)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
