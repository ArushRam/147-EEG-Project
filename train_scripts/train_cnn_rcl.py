from networks.CNN_RCL import CNN_RCL
from util.functions import set_all_seeds
from util.loaders import get_loaders
from util.trainer import Trainer
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# SET SEEDS
set_all_seeds(seed=0)

# IMPORT NETWORK

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
    'trim_size': 500,
}

input_size = (22, 1, 250)
num_classes = 4
activation = nn.ELU
model = CNN_RCL


def run():
    # Initialize Model
    model_instance = model().float()
    # Get Loaders
    loaders = get_loaders(
        batch_size=hyperparameters['batch_size'], preprocess_params=preprocess_params)
    print(loaders)
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
