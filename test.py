import torch
import torch.nn as nn
import torch.optim as optim

# Define the model
class FewShotModel(nn.Module):
    def __init__(self, feature_size, num_classes):
        super().__init__()
        self.fc = nn.Linear(feature_size, num_classes)
        
    def forward(self, x):
        x = self.fc(x)
        return x

# Create the model
feature_size = 64
num_classes = 5
model = FewShotModel(feature_size, num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Generate the mock train_query_set tensor
feature_size = 64
num_examples = 10
train_query_set = torch.randn(num_examples, feature_size)

# Generate the mock train_labels tensor
num_classes = 5
train_labels = torch.randint(0, num_classes, (num_examples,))

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    # Zero the gradients
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(train_query_set)
    loss = criterion(outputs, train_labels)
    
    # Backward pass
    loss.backward()
    
    # Update the parameters
    optimizer.step()
    
    # Print the loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
