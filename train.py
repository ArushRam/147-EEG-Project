from networks.CNN import BasicCNN
from dataset import EEGDataPreprocessor, EEGDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import to_categorical
from torch.nn.functional import one_hot

# HYPERPARAMETERS
num_epochs = 50
batch_size = 64
learning_rate = 0.001

processed_data = EEGDataPreprocessor()
train_dataset = EEGDataset(processed_data.x_train, processed_data.y_train)
val_dataset = EEGDataset(processed_data.x_valid, processed_data.y_valid)
test_dataset = EEGDataset(processed_data.x_test, processed_data.y_test)

# Create dataloaders for training, validation, and testing data
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

### MODEL INITIALIZATION ###
# Define Architecture
conv_params = [
    {'kernel_size': (1, 10), 'num_filters': 25, 'padding': (0, 5)},
    {'kernel_size': (1, 10), 'num_filters': 50, 'padding': (0, 5)},
    {'kernel_size': (1, 10), 'num_filters': 100, 'padding': (0, 5)},
    {'kernel_size': (1, 10), 'num_filters': 200, 'padding': (0, 5)}
]
pool_params = {'kernel_size': (1, 3), 'padding': (0, 1)}
input_size = (22, 1, 250)
num_classes = 4

# Initialize Model
model = BasicCNN(input_size, num_classes, conv_params, pool_params).double()
print(model)

# Define the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    # Set the model to training mode
    model.train()

    # Loop over the batches in the dataset
    for batch_idx, (data, target) in enumerate(train_loader):
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)

        # Compute the loss
        loss = loss_fn(output, target.float())

        # Backward pass
        loss.backward()

        # Update the parameters
        optimizer.step()

        # Print the progress
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    # Evaluate the model on the validation set
    model.eval()
    val_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            val_loss += loss_fn(output, target.float()).item()
            pred = output.argmax(dim=1, keepdim=True)
            target = target.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    accuracy = 100. * correct / len(val_loader.dataset)
    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset), accuracy))
