import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
import datetime
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks.ST_CNN import ShallowCNN
from dataset import EEGDataPreprocessor, EEGDataset
from util.functions import to_categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# HYPERPARAMETERS
num_epochs = 50
batch_size = 128
learning_rate = 0.001
writer = SummaryWriter()

processed_data = EEGDataPreprocessor()
# Swap axes for ST-CNN layour
train_dataset = EEGDataset(np.swapaxes(processed_data.x_train, 1, 2), processed_data.y_train)
val_dataset = EEGDataset(np.swapaxes(processed_data.x_valid, 1, 2), processed_data.y_valid)
test_dataset = EEGDataset(np.swapaxes(processed_data.x_test, 1, 2), processed_data.y_test)

# Create dataloaders for training, validation, and testing data
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

datetime_str = datetime.datetime.now().strftime("%Y%m%d-%H%M")
model_save_dir = 'logs/' + datetime_str + '/model'

### MODEL INITIALIZATION ###
input_size = (1, 22, 250)
num_classes = 4
model = ShallowCNN(input_size, num_classes).float()
print(model)

# Device Config
device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
elif torch.backends.mps.is_available():
    device = "mps"
print("Device: ", device)

model.to(device)

# Define the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)

for epoch in range(num_epochs):
    # Set the model to training mode
    model.train()

    train_loss = 0
    # Loop over the batches in the dataset
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target.float())
        train_loss += loss
        loss.backward()
        optimizer.step()

        # Print progress
        # if batch_idx % 10 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))

    # Evaluate the model on the validation set
    train_loss /= len(train_loader.dataset)
    writer.add_scalar("Train_Loss", train_loss, epoch)
    model.eval()
    val_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += loss_fn(output, target.float()).item()
            pred = output.argmax(dim=1, keepdim=True)
            target = target.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    
    writer.add_scalar("Valid_Loss", val_loss, epoch)
    accuracy = 100. * correct / len(val_loader.dataset)

    writer.add_scalar("Valid_accuracy", accuracy, epoch)
    writer.flush()
    # print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     val_loss, correct, len(val_loader.dataset), accuracy))
    
    model.save(epoch, optimizer, model_save_dir)

model.eval()

# Initialize variables to keep track of accuracy and loss
test_loss = 0.0
correct = 0

# Iterate over batches of test data

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += loss_fn(output, target.float()).item()
        pred = output.argmax(dim=1, keepdim=True)
        target = target.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)
accuracy = 100. * correct / len(test_loader.dataset)
print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset), accuracy))
