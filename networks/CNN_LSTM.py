import torch
import torch.nn as nn
import os


class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()

        # Conv. block 1

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=22, out_channels=25, kernel_size=(1, 10), padding=(0, 5)),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(
                1, 3), stride=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(25),
            nn.Dropout2d(p=0.5)
        )

        # Conv. block 2
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=25, out_channels=50, kernel_size=(1, 10), padding=(0, 5)),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(
                1, 3), stride=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(50),
            nn.Dropout2d(p=0.5)
        )

        # Conv. block 3
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=50, out_channels=100, kernel_size=(1, 10), padding=(0, 5)),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(
                1, 3), stride=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(100),
            nn.Dropout2d(p=0.5)
        )

        # Conv. block 4
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(
                in_channels=100, out_channels=200, kernel_size=(1, 10), padding=(0, 5)),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(
                1, 3), stride=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(200),
            nn.Dropout2d(p=0.5)
        )

        # FC+LSTM layers
        self.ltsm_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=200*4, out_features=200),
            nn.LSTM(input_size=200, hidden_size=100, num_layers=1,
                    batch_first=True, bidirectional=False),

        )

        self.ltsm_2 = nn.Sequential(
            nn.LSTM(input_size=100, hidden_size=50, num_layers=1,
                    batch_first=True, bidirectional=False),

        )

        self.output = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(in_features=50, out_features=4),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x, _ = self.ltsm_1(x)
        x, _ = self.ltsm_2(x)
        x = self.output(x)

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
