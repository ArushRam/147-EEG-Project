import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
import datetime
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks.CNN import BasicCNN
from dataset import EEGDataPreprocessor, EEGDataset
from util.functions import to_categorical
from tensorboardX import SummaryWriter
from tqdm import tqdm

# HYPERPARAMETERS
num_epochs = 200
batch_size = 64
learning_rate = 0.0001
writer = SummaryWriter()

processed_data = EEGDataPreprocessor()
train_dataset = EEGDataset(processed_data.x_train, processed_data.y_train)
val_dataset = EEGDataset(processed_data.x_valid, processed_data.y_valid)
test_dataset = EEGDataset(processed_data.x_test, processed_data.y_test)

# Create dataloaders for training, validation, and testing data
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

datetime_str = datetime.datetime.now().strftime("%Y%m%d-%H%M")
model_save_dir = 'logs/' + datetime_str + '/model'

### MODEL INITIALIZATION ###
# Define Architecture
conv_params = [
    {'kernel_size': (1, 10), 'num_filters': 25, 'padding': (0, 5)},
    {'kernel_size': (1, 10), 'num_filters': 50, 'padding': (0, 5)},
    {'kernel_size': (1, 10), 'num_filters': 100, 'padding': (0, 5)},
    {'kernel_size': (1, 10), 'num_filters': 200, 'padding': (0, 5)}
]
pool_params = {'pool_fn': nn.AvgPool2d, 'kernel_size': (1, 3), 'padding': (0, 1)}
input_size = (22, 1, 250)
num_classes = 4

# Initialize Model
model = BasicCNN(input_size, num_classes, conv_params, pool_params)
print(model)

# Define the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    # Set the model to training mode
    model.train()

    train_loss = 0
    # Loop over the batches in the dataset
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
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
    writer.add_scalar("Train Loss", train_loss, epoch)
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
    
    writer.add_scalar("Valid Loss", val_loss, epoch)
    accuracy = 100. * correct / len(val_loader.dataset)

    writer.add_scalar("Valid accuracy", accuracy, epoch)
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
        output = model(data)
        test_loss += loss_fn(output, target.float()).item()
        pred = output.argmax(dim=1, keepdim=True)
        target = target.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)
accuracy = 100. * correct / len(test_loader.dataset)
print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset), accuracy))
