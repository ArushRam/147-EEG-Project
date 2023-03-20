import torch
import torch.nn as nn
import os


class CNN_RCL(nn.Module):
    def __init__(self):
        super(CNN_RCL, self).__init__()

        self.conv = nn.Conv2d(in_channels=22, out_channels=256, kernel_size=(
            1, 9), stride=(1, 1), padding=(0, 4))
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 1))
        self.rcl1 = RCLLayer(256, 256, 9, 3)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 1))
        self.rcl2 = RCLLayer(256, 256, 9, 3)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 1))
        self.rcl3 = RCLLayer(256, 256, 9, 3)
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 1))
        self.rcl4 = RCLLayer(256, 256, 9, 3)
        self.pool5 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 1))
        self.fc = nn.Linear(60672, 4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)
        x = self.rcl1(x)
        x = self.pool2(x)
        x = self.rcl2(x)
        x = self.pool3(x)
        x = self.rcl3(x)
        x = self.pool4(x)
        x = self.rcl4(x)
        x = self.pool5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)

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


class RCLLayer(nn.Module):
    def __init__(self, filters, in_channels, kernel_size, iterations):
        super(RCLLayer, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(in_channels=in_channels,
                           out_channels=filters, kernel_size=(1, 1), stride=(1, 1)))
        self.layers.append(nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=(
            1, kernel_size), stride=(1, 1), padding=(0, kernel_size//2)))
        self.layers.append(nn.ELU())
        for i in range(iterations - 1):
            self.layers.append(nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=(
                1, kernel_size), stride=(1, 1), padding=(0, kernel_size//2)))
            self.layers.append(nn.ELU())

    def forward(self, x):
        residual = x
        for layer in self.layers:
            x = layer(x)
        x += residual
        x = nn.functional.relu(x)
        return x
