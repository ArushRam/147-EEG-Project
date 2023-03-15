import torch.nn as nn
from utils import Bandpass
import torch
import os

class FourierCNN(nn.Module):
    def __init__(self, num_bins, sample_freq, device, num_classes=4):
        super(FourierCNN, self).__init__()

        self.temporal_filters = []
        
        for i in range(8, 28, 2):
            self.temporal_filters.append(
                nn.Sequential(
                    Bandpass(i, i+4, device, num_bins=num_bins, sample_hz=sample_freq),
                nn.Conv2d(in_channels=22, out_channels=10,
                        kernel_size=(1, 10), padding=(0, 5)).to(device),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 3), padding=(0, 1)),
                nn.BatchNorm2d(10).to(device),
                nn.Dropout2d(p=0.5)
                )
            )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=84 * 100, out_features=4),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # print("Input shape", x.shape)
        x = [temporal_filter(x) for temporal_filter in self.temporal_filters]

        # print("Temporal filter output shape:", x[0].shape)
        x = torch.cat(x, dim=1)
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
