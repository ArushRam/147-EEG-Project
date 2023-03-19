import torch.nn as nn
import torch
import os

class Inception1(nn.Module):
    def __init__(self,
        input_size=(1, 22, 250),
        num_classes=4,
        conv_params=None,
        pool_params=None,
        activation=nn.ELU
    ):
        super(Inception1, self).__init__()
        if not conv_params:
            raise ValueError("conv_params cannot be NoneType")
        if not pool_params:
            raise ValueError("pool_params cannot be NoneType")
        
        dims = (input_size[1], input_size[2])

        self.temporal_convs = nn.ModuleList()
        n_filters = 0
        for i in range(3):
            pad = (conv_params[f'k{i+1}'] - 1)//2
            n_filters += conv_params[f'n{i+1}']
            self.temporal_convs.append(nn.Sequential(
                nn.Conv2d(1, conv_params[f'n{i+1}'], (1,conv_params[f'k{i+1}']), padding=(0,pad), bias=False),
                nn.BatchNorm2d(conv_params[f'n{i+1}']),
                activation(),
                nn.Dropout2d(p=0.5)
            ))

        self.pool = nn.AvgPool2d(kernel_size=(1,pool_params['k']), stride=(1,pool_params['s']), padding=(0, 1))
        dims = (1, dims[1] - conv_params['k1'] + 2 * (conv_params['k1'] - 1)//2 + 1)
        print(dims)
        dims = (1, (dims[1] - pool_params['k'] + 2)//pool_params['s'] + 1)
        print(dims)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels=n_filters, out_channels=conv_params['ns'], kernel_size=(input_size[1],1), padding=(0,0)),
            nn.BatchNorm2d(conv_params['ns']),
            activation(),
            nn.Dropout2d(p=0.5)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_params['ns'] * dims[0] * dims[1], num_classes),
            nn.Softmax(dim=1)
        )


    def forward(self, x):
        z = [conv(x) for conv in self.temporal_convs]
        z = torch.cat(z, dim=1)
        z = self.pool(z)
        z = self.spatial_conv(z)
        z = self.fc(z)
        return z

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