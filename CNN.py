import torch.nn as nn


class BasicCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(BasicCNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=22, out_channels=25,
                      kernel_size=(1, 10), padding=(0, 5)),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(25),
            nn.Dropout2d(p=0.5)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=25, out_channels=50,
                      kernel_size=(1, 10), padding=(0, 5)),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(50),
            nn.Dropout2d(p=0.5)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=50, out_channels=100,
                      kernel_size=(1, 10), padding=(0, 5)),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(100),
            nn.Dropout2d(p=0.5)
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=100, out_channels=200,
                      kernel_size=(1, 10), padding=(0, 5)),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(200),
            nn.Dropout2d(p=0.5)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=200 * 4, out_features=4),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        # print("Block 1 output shape:", x.shape)
        x = self.conv_block2(x)
        # print("Block 2 output shape:", x.shape)
        x = self.conv_block3(x)
        # print("Block 3 output shape:", x.shape)
        x = self.conv_block4(x)
        # print("Block 4 output shape:", x.shape)
        x = self.fc(x)
        # print("FC output shape:", x.shape)
        return x
