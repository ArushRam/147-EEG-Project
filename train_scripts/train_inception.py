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

k = [1, 5, 7, 11]#, 17]
n0 = 48
layers = 3
n = 48
bp = (0.5, 30)
run_name = f'Inception_k=[{k[:]}]_bp={bp}_n={n})'

hyperparameters = {
    'loss_fn': nn.CrossEntropyLoss(),
    'optimizer': optim.Adam,
    'num_epochs': 100,
    'learning_rate': 0.001,
    'weight_decay': 0.01,
    'batch_size': 128,
}

preprocess_params = {
    'valid_ratio': 0.2,
    'trim_size': 512,
    'maxpool': True,
    'sub_sample': 4,
    'average': True,
    'bp_range': bp,
    'crop': 120,
    'noise': 0.5,
}
### MODEL INITIALIZATION ###
# Define Architecture
conv_params = {
    'layers': layers,
    'k0': k[0], 'n0': n0,
    'k1': k[1], 'n1': n,
    'k2': k[2], 'n2': n,
    'k3': k[3], 'n3': n,
#    'k4': k[4], 'n4': n,
#     'k5': k[5], 'n5': n,
}
pool_params = {
    'pool_fn': nn.MaxPool2d, 'k': 7, 's': 3
}

input_size = (22, 1, preprocess_params['crop'])
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

    # Train Model
    trainer.train()
    # Evaluate Model
    trainer.evaluate(mode='test')
    # Get Stats
    trainer.print_stats()
    return

run()

