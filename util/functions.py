import numpy as np
import torch.nn as nn
import torch.fft as fft
import torch


class Bandpass(nn.Module):
    def __init__(self, lower, upper, device=None, num_bins=1000, sample_hz=250) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.sample_hz = sample_hz
        freq = fft.fftfreq(self.num_bins, 1/self.sample_hz)
        self.mask = (freq >= lower) * (freq <= upper)
        if device is not None:
            self.mask = self.mask.to(device)
        
    def forward(self, x):
        f_x = fft.fft(x, dim=-1)
        f_x = self.mask * f_x
        return torch.real(fft.ifft(f_x))


class FreqDomain(nn.Module):
    def __init__(self, lower, upper, device=None, num_bins=1000, sample_hz=250) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.sample_hz = sample_hz
        freq = fft.fftfreq(self.num_bins, 1/self.sample_hz)
        self.mask = (freq >= lower) * (freq <= upper)
        if device is not None:
            self.mask = self.mask.to(device)
        
    def forward(self, x):
        f_x = fft.fft(x, dim=-1)
        f_x = self.mask * f_x
        return torch.real(fft.ifft(f_x))


def data_prep(X,y,sub_sample,average,noise):
    
    total_X = None
    total_y = None
    
    # Trimming the data (sample,22,1000) -> (sample,22,500)
    X = X[:,:,0:500]
    print('Shape of X after trimming:',X.shape)
    
    # Maxpooling the data (sample,22,1000) -> (sample,22,500/sub_sample)
    X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, sub_sample), axis=3)
    
    
    total_X = X_max
    total_y = y
    print('Shape of X after maxpooling:',total_X.shape)
    
    # Averaging + noise 
    X_average = np.mean(X.reshape(X.shape[0], X.shape[1], -1, average),axis=3)
    X_average = X_average + np.random.normal(0.0, 0.5, X_average.shape)
    
    total_X = np.vstack((total_X, X_average))
    total_y = np.hstack((total_y, y))
    print('Shape of X after averaging+noise and concatenating:',total_X.shape)
    
    # Subsampling
    
    for i in range(sub_sample):
        
        X_subsample = X[:, :, i::sub_sample] + \
                            (np.random.normal(0.0, 0.5, X[:, :,i::sub_sample].shape) if noise else 0.0)
            
        total_X = np.vstack((total_X, X_subsample))
        total_y = np.hstack((total_y, y))
        
    
    print('Shape of X after subsampling and concatenating:',total_X.shape)
    return total_X,total_y


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]