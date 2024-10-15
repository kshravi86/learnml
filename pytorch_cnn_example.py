import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Device configuration
# Use GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_epochs = 5  # Number of times the entire dataset is passed through the network
batch_size = 64  # Number of samples per gradient update
learning_rate = 0.001  # Step size for the optimizer

# Load MNIST dataset
# Transform to tensor and normalize the data (mean=0.5, std=0.5)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download and load training and test datasets
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create data loaders for training and testing
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define a simple CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # First convolutional layer: input channel = 1, output channel = 16, kernel size = 3x3
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        # Second convolutional layer: input channel = 16, output channel = 32, kernel size = 3x3
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # Max pooling layer: kernel size = 2x2, stride = 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # Fully connected layer: input size = 32 * 7 * 7, output size = 128
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        # Fully connected layer: input size = 128, output size = 10 (number of classes)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Pass input through the first convolutional layer, apply ReLU, then max pool
        x = self.pool(F.relu(self.conv1(x)))
        # Pass input through the second convolutional layer, apply ReLU, then max pool
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the output for the fully connected layers
        x = x.view(-1, 32 * 7 * 7)
        # Pass through the first fully connected layer and apply ReLU
        x = F.relu(self.fc1(x))
        # Pass through the second fully connected layer (no activation, as this is the final output)
        x = self.fc2(x)
        return x

# Instantiate the model, define loss function and optimizer
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification tasks
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        # Move images and labels to the configured device (CPU or GPU)
        images, labels = images.to(device), labels.to(device)

        # Forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(images)
        # Calculate the loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()  # Clear existing gradients for all model parameters
        loss.backward()  # Compute gradient of loss w.r.t model parameters
        optimizer.step()  # Update model parameters

        running_loss += loss.item()  # Accumulate loss
        # Print loss every 100 mini-batches
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# Testing loop
model.eval()  # Set model to evaluation mode
with torch.no_grad():  # Disable gradient calculation for inference
    correct = 0
    total = 0
    for images, labels in test_loader:
        # Move images and labels to the configured device
        images, labels = images.to(device), labels.to(device)
        # Forward pass: compute predicted outputs
        outputs = model(images)
        # Get the class with the highest probability
        _, predicted = torch.max(outputs.data, 1)
        # Update total number of samples and number of correct predictions
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Print the accuracy of the model on the test dataset
    print(f'Accuracy of the model on the 10000 test images: {100 * correct / total} %')

# Save the model checkpoint
torch.save(model.state_dict(), 'cnn_model.pth')  # Save the trained model's state dictionary
