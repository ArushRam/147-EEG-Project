import torch.nn as nn


class BasicCNN(nn.Module):
    def __init__(self, 
        input_size=(22, 1, 250),
        num_classes=4, 
        conv_params=None,
        pool_params=None,
        activation=nn.ELU
    ):
        '''
        Create a simple CNN with block structure.

        Keyword Arguments:
            input_size -- dimensions of single input instance
            num_classes -- default 4 (won't change)
            conv_params -- a list of dictionaries with 'kernel_size', 'num_filters', 'padding' (optional), 'stride' (optional)
            pool_params -- a dictionary or list of dictionaries with 'kernel_size', 'padding' (optional), 'stride' (optional)
            activation -- activation function used in conv layers, default is nn.ELU (exponential linear unit) 
        '''

        super(BasicCNN, self).__init__()

        if not pool_params or not conv_params:
            raise ValueError("conv_params and pool_params cannot be NoneType")
        self.n_blocks = len(conv_params)
        if isinstance(pool_params, list) and len(pool_params) != self.n_blocks:
            raise ValueError('Length of pool_params does not match length of conv_params')
        elif isinstance(pool_params, dict):
            pool_params = [pool_params] * self.n_blocks

        in_channels = input_size[0]
        dims = input_size[1:]
        print(dims)
                
        self.conv_blocks = [None] * self.n_blocks
        channels = in_channels

        for i in range(self.n_blocks):
            kernel_size, num_filters = conv_params[i]['kernel_size'], conv_params[i]['num_filters']
            conv_stride, conv_padding = conv_params[i].get('stride', (1, 1)), conv_params[i].get('padding', (0, 0))
            pool_size = pool_params[i]['kernel_size']
            pool_stride, pool_padding = pool_params[i].get('stride', pool_size), pool_params[i].get('padding', (0, 0))
            self.conv_blocks[i] = nn.Sequential(
                nn.Conv2d(channels, num_filters, kernel_size, conv_stride, conv_padding),
                activation(),
                nn.MaxPool2d(pool_size, pool_stride, pool_padding),
                nn.BatchNorm2d(num_filters),
                nn.Dropout2d(p=0.5)
            )
            channels = num_filters
            dims = [(dims[j] - kernel_size[j] + 2 * conv_padding[j])//conv_stride[j] + 1 for j in range(2)]
            dims = [(dims[j] - pool_size[j] + 2 * pool_padding[j])//pool_stride[j] + 1 for j in range(2)]
            print(dims)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=channels * dims[0] * dims[1], out_features=num_classes),
            nn.Softmax(dim=1)
        )

        # self.conv_block1 = nn.Sequential(
        #     nn.Conv2d(in_channels=22, out_channels=25,
        #               kernel_size=(1, 10), padding=(0, 5)),
        #     nn.ELU(),
        #     nn.MaxPool2d(kernel_size=(1, 3), padding=(0, 1)),
        #     nn.BatchNorm2d(25),
        #     nn.Dropout2d(p=0.5)
        # )

        # self.conv_block2 = nn.Sequential(
        #     nn.Conv2d(in_channels=25, out_channels=50,
        #               kernel_size=(1, 10), padding=(0, 5)),
        #     nn.ELU(),
        #     nn.MaxPool2d(kernel_size=(1, 3), padding=(0, 1)),
        #     nn.BatchNorm2d(50),
        #     nn.Dropout2d(p=0.5)
        # )

        # self.conv_block3 = nn.Sequential(
        #     nn.Conv2d(in_channels=50, out_channels=100,
        #               kernel_size=(1, 10), padding=(0, 5)),
        #     nn.ELU(),
        #     nn.MaxPool2d(kernel_size=(1, 3), padding=(0, 1)),
        #     nn.BatchNorm2d(100),
        #     nn.Dropout2d(p=0.5)
        # )

        # self.conv_block4 = nn.Sequential(
        #     nn.Conv2d(in_channels=100, out_channels=200,
        #               kernel_size=(1, 10), padding=(0, 5)),
        #     nn.ELU(),
        #     nn.MaxPool2d(kernel_size=(1, 3), padding=(0, 1)),
        #     nn.BatchNorm2d(200),
        #     nn.Dropout2d(p=0.5)
        # )

        # self.fc = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(in_features=200 * 4, out_features=4),
        #     nn.Softmax(dim=1)
        # )

    def forward(self, x):
        for i in range(self.n_blocks):
            x = self.conv_blocks[i](x)
            print(f'Block {i} output shape: {x.shape}')
        x = self.fc(x)
        print(f'Output shape: {x.shape}')
        return x
    
    # def forward(self, x):
    #     x = self.conv_block1(x)
    #     # print("Block 1 output shape:", x.shape)
    #     x = self.conv_block2(x)
    #     # print("Block 2 output shape:", x.shape)
    #     x = self.conv_block3(x)
    #     # print("Block 3 output shape:", x.shape)
    #     x = self.conv_block4(x)
    #     # print("Block 4 output shape:", x.shape)
    #     x = self.fc(x)
    #     # print("FC output shape:", x.shape)
    #     return x
