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
from networks.ST_CNN import ShallowCNN

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
    'swap_axes': (1, 2),
}

### MODEL INITIALIZATION ###
input_size = (1, 22, 250)
num_classes = 4
activation = nn.ELU
params = (input_size, num_classes, activation)
model = ShallowCNN

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