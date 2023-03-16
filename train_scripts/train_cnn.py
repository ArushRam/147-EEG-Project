import torch
import torch.nn as nn
import torch.optim as optim
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.trainer import Trainer
from util.loaders import get_loaders
from util.functions import set_all_seeds

# SET SEEDS
set_all_seeds(seed=0)

# IMPORT NETWORK
from networks.ST_CNN import BasicCNN

# HYPERPARAMETERS
hyperparameters = {
    'loss_fn': nn.CrossEntropyLoss(),
    'optimizer': optim.Adam,
    'num_epochs': 10,
    'learning_rate': 0.0001,
    'weight_decay': 0.001,
    'batch_size': 128,
}

preprocess_params = {
#   'valid_ratio': 0.2,
    'trim_size': 250,
}

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
activation = nn.ELU
## WRITE PARAMS INTO TUPLE SO YOU CAN USE THE MAIN FUNCTION WITHOUT CHANGES
params = (input_size, num_classes, conv_params, pool_params, activation)
model = BasicCNN

def run():
    # Initialize Model
    model_instance = model(*params).float()
    # Get Loaders
    loaders = get_loaders(batch_size=hyperparameters['batch_size'], preprocess_params=preprocess_params)
    # Initialize Trainer
    trainer = Trainer(loaders, model_instance, hyperparameters)
    # Train Model
    trainer.train()
    # Evaluate Model
    trainer.evaluate(mode='test')
    # Get Stats
    trainer.print_stats()
    return

run()

