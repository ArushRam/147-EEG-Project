import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
from tqdm import tqdm
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks.Fourier_CNN import FourierCNN
from networks.Fourier_CNN2 import FourierCNN as LinearFourier
from dataset import EEGDataPreprocessor, EEGDataset
from util.functions import to_categorical
from tensorboardX import SummaryWriter
import datetime

processed_data = EEGDataPreprocessor()


train_dataset = EEGDataset(processed_data.x_train, processed_data.y_train)
val_dataset = EEGDataset(processed_data.x_valid, processed_data.y_valid)
test_dataset = EEGDataset(processed_data.x_test, processed_data.y_test)

# HYPERPARAMETERS
num_epochs = 500
batch_size = 64
learning_rate = 0.0001
writer = SummaryWriter()

# Create dataloaders for training, validation, and testing data
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

datetime_str = datetime.datetime.now().strftime("%Y%m%d-%H%M")
model_save_dir = 'logs/' + datetime_str + '/model'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BasicCNNModel = FourierCNN(sample_freq=250/4, num_bins=250, device=device)

BasicCNNModel.to(device)


# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(BasicCNNModel.parameters(), lr=learning_rate, weight_decay=1e-1)


# print(next(BasicCNNModel.parameters()).is_cuda)

for epoch in range(num_epochs):
    # Set the model to training mode
    BasicCNNModel.train()

    train_loss = 0
    # Loop over the batches in the dataset
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        # Zero the gradients
        optimizer.zero_grad()

        data, target = data.to(device), target.to(device)
        
        # Forward pass
        output = BasicCNNModel(data)

        # Compute the loss
        loss = criterion(output, target.float())
        train_loss += loss
        # Backward pass
        loss.backward()

        # Update the parameters
        optimizer.step()

        # Print the progress
        # if batch_idx % 1000 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))

    # Evaluate the model on the validation set
    train_loss /= len(train_loader.dataset)
    writer.add_scalar("Train Loss", train_loss, epoch)
    
    BasicCNNModel.eval()
    val_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = BasicCNNModel(data)
            val_loss += criterion(output, target.float()).item()
            pred = output.argmax(dim=1, keepdim=True)
            target = target.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    writer.add_scalar("Valid Loss", val_loss, epoch)
    accuracy = 100. * correct / len(val_loader.dataset)
    
    writer.add_scalar("Valid accuracy", accuracy, epoch)
    writer.flush()
    # print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    # val_loss, correct, len(val_loader.dataset), accuracy))
    BasicCNNModel.save(epoch, optimizer, model_save_dir)

BasicCNNModel.eval()

# Initialize variables to keep track of accuracy and loss
test_loss = 0.0
correct = 0

# Iterate over batches of test data

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = BasicCNNModel(data)
        test_loss += criterion(output, target.float()).item()
        pred = output.argmax(dim=1, keepdim=True)
        target = target.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)
accuracy = 100. * correct / len(test_loader.dataset)
print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset), accuracy))
