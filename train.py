from CNN import BasicCNN
from dataset import EEGDataPreprocessor, EEGDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import to_categorical
from torch.nn.functional import one_hot
import datetime

processed_data = EEGDataPreprocessor()


train_dataset = EEGDataset(processed_data.x_train, processed_data.y_train)
val_dataset = EEGDataset(processed_data.x_valid, processed_data.y_valid)
test_dataset = EEGDataset(processed_data.x_test, processed_data.y_test)

# Define batch size for training and testing
batch_size = 64

# Create dataloaders for training, validation, and testing data
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

datetime_str = datetime.datetime.now().strftime("%Y%m%d-%H%M")
model_save_dir = 'logs/' + datetime_str + '/model'

BasicCNNModel = BasicCNN()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(BasicCNNModel.parameters(), lr=0.0001)

# Train the model
num_epochs = 50

for epoch in range(num_epochs):
    # Set the model to training mode
    BasicCNNModel.train()

    # Loop over the batches in the dataset
    for batch_idx, (data, target) in enumerate(train_loader):
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = BasicCNNModel(data)

        # Compute the loss
        loss = criterion(output, target.float())

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
    BasicCNNModel.eval()
    val_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in val_loader:
            output = BasicCNNModel(data)
            val_loss += criterion(output, target.float()).item()
            pred = output.argmax(dim=1, keepdim=True)
            target = target.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    accuracy = 100. * correct / len(val_loader.dataset)
    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset), accuracy))
    
    BasicCNNModel.save(epoch, optimizer, model_save_dir)

BasicCNNModel.eval()

# Initialize variables to keep track of accuracy and loss
test_loss = 0.0
correct = 0

# Iterate over batches of test data

with torch.no_grad():
    for data, target in test_loader:
        output = BasicCNNModel(data)
        test_loss += criterion(output, target.float()).item()
        pred = output.argmax(dim=1, keepdim=True)
        target = target.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)
accuracy = 100. * correct / len(test_loader.dataset)
print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset), accuracy))
