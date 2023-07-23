import numpy as np
import torch.nn as nn
import torch.fft as fft
import torch
import random
from scipy import signal

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


def data_prep(X , y, person, params, mode='train'):
    
    trim_size = params.get('trim_size', 500)
    maxpool = params.get('maxpool', True)
    sub_sample = params.get('sub_sample', 1)
    average = params.get('average', True)
    noise = params.get('noise', True)
    bp_range = params.get('bp_range', None)
    mean, std = params.get('stats', None)
    if mode == 'test':
        noise = 0
    total_X = None
    total_y = None
    
    # Trimming the data (sample,22,1000) -> (sample,22,500)
    X = X[:,:,0:trim_size]
    print('Shape of X after trimming:',X.shape)

    # Z3 based normalization from the paper
    # mean, std = np.mean(X), np.std(X)
    # X = (X - mean)/std
    # mean, std = np.mean(X, axis=2), np.std(X, axis=2)
    # X = (X - mean[:,:,np.newaxis])/std[:,:,np.newaxis]

    # if maxpool:
    #     # Maxpooling the data (sample,22,1000) -> (sample,22,500/sub_sample)
    #     X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, sub_sample), axis=3)
    #     total_X = X_max
    #     total_y = y
    # #     print('Shape of X after maxpooling:',total_X.shape)

    # if average:
    #     # Averaging + noise 
    #     X_average = np.mean(X.reshape(X.shape[0], X.shape[1], -1, sub_sample),axis=3)
    #     if noise:
    #         X_average = X_average + np.random.normal(0.0, noise, X_average.shape)
    #     if total_X is None:
    #         total_X = X_average
    #         total_y = y
    #     else:
    #         total_X = np.vstack((total_X, X_average))
    #         total_y = np.hstack((total_y, y))
    #     print('Shape of X after averaging+noise and concatenating:',total_X.shape)
    
    # # Subsampling
    # for i in range(sub_sample):
        
    #     X_subsample = X[:, :, i::sub_sample] # + \
    #                         # (np.random.normal(0.0, 0.25, X[:, :,i::sub_sample].shape) if noise else 0.0)
    #     if total_X is None:
    #         total_X = X_subsample
    #         total_y = y
    #     else:
    #         total_X = np.vstack((total_X, X_subsample))
    #         total_y = np.hstack((total_y, y))
    #     print('Shape of X after subsampling and concatenating:',total_X.shape)

    # # BUTTERWORTH FILTER
    # if bp_range:
    #     sos = signal.butter(5, bp_range, 'bp', fs=250/sub_sample, output='sos')
    #     total_X = signal.sosfilt(sos, total_X, axis=-1)

    if mode == 'train':
        new_X, new_y, new_persons = noise_mixing_augmentation(X, y, person)
        return new_X, new_y
    
    return X, y

    # print('Final X Shape: ', total_X.shape)
    # return total_X,total_y

def noise_mixing_augmentation(X, y, persons, noise_threshold=100):
    unique_labels = np.unique(y)
    unique_persons = np.unique(persons)
    new_Xs, new_ys, new_persons_list = [], [], []
    for label in unique_labels:
        for person in unique_persons:
            idx = np.logical_and(y == label, persons == person)

            print(X[:, idx].shape)

            # Extract noise for each X in this class
            sos_noise = signal.butter(8, noise_threshold, btype='highpass', output='sos', fs=250)
            noise = signal.sosfilt(sos_noise, X[idx, :], axis=-1)

            # Signal candidates
            signals = X[idx, :] - noise

            N = len(X[idx, :])

            new_Xs.append(np.stack([signals[i] + noise[j] for i in range(N) for j in range(N)]))
            new_ys.append(np.full((len(new_Xs[-1])), fill_value=label))
            new_persons_list.append(np.full((len(new_Xs[-1])), fill_value=person))

    new_X = np.concatenate(new_Xs)
    new_y = np.concatenate(new_ys)
    new_persons_list = np.concatenate(new_persons_list)

    return new_X, new_y, new_persons_list

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True