import torch.nn as nn
import torch
import os

class ShallowCNN(nn.Module):
    '''
    Implements the Shallow CNN architecture as proposed by Schirrmeister et al, 2018 [https://arxiv.org/pdf/1703.05051.pdf].
    Architecture:
        - Temporal Convolution
        - Spatial Convolution across ALL electrodes
        - AvgPool
        - FC + Softmax

    !!! Takes in data formatted as (1, 22, 250) as opposed to (22, 1, 250) !!!

    Will implement generic one with variable parameters if necessary
    '''
    def __init__(self,
        input_size=(1, 22, 250),
        num_classes=4,
        activation=nn.ELU()
    ):
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=40, kernel_size=(1, 25))
        self.conv2 = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=(22, 1))
        self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
