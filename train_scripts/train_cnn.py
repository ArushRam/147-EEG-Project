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
from networks.CNN import BasicCNN

# HYPERPARAMETERS

# k1, k2 = 15, 5
k = 5
nfilters = 40
psize = 7
pstride = 3
run_name = f'CNN1-k{k}-f{nfilters}-p{psize}-ps{pstride}(125*6)(0.0001)'

hyperparameters = {
    'loss_fn': nn.CrossEntropyLoss(),
    'optimizer': optim.Adam,
    'num_epochs': 200,
    'learning_rate': 0.0001,
    'weight_decay': 0.01,
    'batch_size': 128,
}

preprocess_params = {
#   'valid_ratio': 0.2,
    'trim_size': 750,
    'maxpool': True,
    'sub_sample': 6,
    'average': 6
}

### MODEL INITIALIZATION ###
# Define Architecture
conv_params = [
    {'kernel_size': (1, k), 'num_filters': nfilters, 'padding': (0, 0)},
    # {'kernel_size': (1, 3), 'num_filters': nfilters, 'padding': (0, 3)},
    # {'kernel_size': (1, 10), 'num_filters': 25, 'padding': (0, 5)},
    # {'kernel_size': (1, 10), 'num_filters': 25, 'padding': (0, 5)}
]
pool_params = [
    {'pool_fn': nn.AvgPool2d, 'kernel_size': (1, psize), 'padding': (0, 1), 'stride':(1, pstride)},
    # {'pool_fn': nn.AvgPool2d, 'kernel_size': (1, psize), 'padding': (0, 1), 'stride':(1, pstride)},
]
input_size = (22, 1, preprocess_params['trim_size']//preprocess_params['sub_sample'])
num_classes = 4
activation = nn.ELU
## WRITE PARAMS INTO TUPLE SO YOU CAN USE THE MAIN FUNCTION WITHOUT CHANGES
params = (input_size, num_classes, conv_params, pool_params, activation)
model = BasicCNN


def run():
    # Initialize Model
    model_instance = model(*params).float()
    print(model_instance)
    print(sum(p.numel() for p in model_instance.parameters() if p.requires_grad))
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

