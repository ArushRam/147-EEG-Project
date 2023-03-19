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
from networks.inception import Inception1 as model

# HYPERPARAMETERS

# k1, k2 = 15, 5
run_name = f'Inception1-Base'

hyperparameters = {
    'loss_fn': nn.CrossEntropyLoss(),
    'optimizer': optim.Adam,
    'num_epochs': 200,
    'learning_rate': 0.001,
    'weight_decay': 0.01,
    'batch_size': 128,
}

preprocess_params = {
#   'valid_ratio': 0.2,
    'trim_size': 500,
    'maxpool': True,
    'sub_sample': 4,
    'average': 4,
    'swap_axes': (1, 2)
}

### MODEL INITIALIZATION ###
# Define Architecture
conv_params = {
    'k1': 1, 'n1': 8,
    'k2': 3, 'n2': 8,
    'k3': 5, 'n3': 8,
    'ns': 8
}
pool_params = {
    'k': 7, 's': 3
}

input_size = (1, 22, preprocess_params['trim_size']//preprocess_params['sub_sample'])
num_classes = 4
activation = nn.ELU
## WRITE PARAMS INTO TUPLE SO YOU CAN USE THE MAIN FUNCTION WITHOUT CHANGES
params = (input_size, num_classes, conv_params, pool_params, activation)


def run():
    # Initialize Model
    model_instance = model(*params).float()
    # Get Loaders
    loaders = get_loaders(batch_size=hyperparameters['batch_size'], preprocess_params=preprocess_params)
    # Initialize Trainer
    trainer = Trainer(loaders, model_instance, hyperparameters, run_name=run_name)

    return

    # Train Model
    trainer.train()
    # Evaluate Model
    trainer.evaluate(mode='test')
    # Get Stats
    trainer.print_stats()
    return

run()

