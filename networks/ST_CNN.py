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
        activation=nn.ELU
    ):
        super(ShallowCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=40, kernel_size=(1,11)),
            activation(),
            nn.Conv2d(in_channels=40, out_channels=40, kernel_size=(22, 1)),
            activation(),
            nn.AvgPool2d(kernel_size=(1,37), stride=(1,7)),
            nn.BatchNorm2d(40),
            nn.Dropout(p=0.5)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(30 * 40, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc(x)
        return x

    def save(self, epoch, optimizer, path):
         os.makedirs(path, exist_ok=True) 
         data = {
             "epoch": epoch,
             "opt_state_dict": optimizer.state_dict(),
             "model_state_dict": self.state_dict()
         }
         torch.save(data, path + '/epoch=%03d.pth' % epoch)
         
    def load(self, model_path, epoch, optimizer):
        checkpoint = torch.load(model_path + '/epoch=%03d.pth' % epoch)
        self.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['opt_state_dict'])
        return optimizer, epoch


