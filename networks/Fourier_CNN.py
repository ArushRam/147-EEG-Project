import torch.nn as nn
from util.functions import Bandpass, FreqDomain
import torch
import os

class FourierCNN(nn.Module):
    def __init__(self, num_bins, sample_freq, device, num_classes=4):
        super(FourierCNN, self).__init__()

        self.temporal_filters = nn.ModuleList([
            nn.Sequential(
                # N x 22 x 1 x 250
                FreqDomain(i, i+4, device, num_bins=num_bins, sample_hz=sample_freq),
                # N x 22 x 2 x 17
                nn.Conv2d(in_channels=22, out_channels=25,
                        kernel_size=(2, 3), padding=(0, 2)).to(device),
                # N x 25 x 1 x 19
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), padding=(0, 1)),
                # N x 25 x 1 x 10
                nn.BatchNorm2d(25).to(device),
                nn.Dropout2d(p=0.5)
            )
            for i in range(0, 9)
        ])

        self.spatial_filter = nn.Sequential(
            # Stack temporal convolutions
            # N x 250 x 1 x 10
            nn.Conv2d(in_channels=225, out_channels=50,
                        kernel_size=(1, 3), padding=(0, 1)).to(device),
            # N x 50 x 1 x 10
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), padding=(0, 0)),
            # N x 50 x 1 x 5
            nn.BatchNorm2d(50).to(device),
            nn.Dropout2d(p=0.5)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=5 * 50, out_features=100),
            # nn.BatchNorm1d(100),
            # nn.ELU(),
            nn.Linear(in_features=100, out_features=4),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # print("Input shape", x.shape)
        x = [l(x) for i, l in enumerate(self.temporal_filters)]

        x = torch.cat(x, dim=1)
        # print(x.shape)

        x = self.spatial_filter(x)
        # print("Cat output shape:", x.shape)
        x = self.fc(x)
        # print("FC output shape:", x.shape)
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
