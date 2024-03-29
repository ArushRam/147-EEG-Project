import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import EEGDataPreprocessor, EEGDataset
import numpy as np
from torch.utils.data import DataLoader

def get_loaders(dataset=EEGDataset, preprocessor=EEGDataPreprocessor, preprocess_params=None, batch_size=128):
    '''
    Get training, validation and test loaders.
    Arguments:
        dataset
        preprocessor
        batch_size -- default is 128
        swap_axes -- tuple of axes to swap where default shape is (N, 22, 1, 250)
    '''
    valid_ratio = preprocess_params.get('valid_ratio', 0.2)
    swap_axes = preprocess_params.get('swap_axes', None)
    processed_data = preprocessor(valid_ratio=valid_ratio, hyperparams=preprocess_params)
    crop = preprocess_params.get('crop', 100)
    noise = preprocess_params.get('noise', 0.5)
    if swap_axes:
        train_dataset = dataset(np.swapaxes(processed_data.x_train, *swap_axes), processed_data.y_train, crop=crop, nois=noise, mode='train')
        val_dataset = dataset(np.swapaxes(processed_data.x_valid, *swap_axes), processed_data.y_valid, crop=crop, noise=noise, mode='test')
        test_dataset = dataset(np.swapaxes(processed_data.x_test, *swap_axes), processed_data.y_test, crop=crop, noise=noise, mode='test')
    else:
        train_dataset = dataset(processed_data.x_train, processed_data.y_train, crop=crop, noise=noise, mode='train')
        val_dataset = dataset(processed_data.x_valid, processed_data.y_valid, crop=crop, noise=noise, mode='test')
        test_dataset = dataset(processed_data.x_test, processed_data.y_test, crop=crop, noise=noise, mode='test')

    # Create dataloaders for training, validation, and testing data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader