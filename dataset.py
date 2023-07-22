import torch
from torch.utils.data import TensorDataset
import numpy as np
from util.functions import data_prep, to_categorical

class EEGDataset(TensorDataset):
    def __init__(self, x, y, crop=100, noise=0.25, mode='train') -> None:
        super().__init__()
        self.mode = mode
        self.crop = crop
        self.noise = noise
        self.xdim = x.shape[-1]
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

        i = (self.xdim - crop)//2
        self.cropidx = (i, self.xdim - i)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        target = self.y[index]
        data_val = self.x[index]
        if self.mode == 'train':
            data_val += torch.normal(0, self.noise, size=(1, self.xdim))
            idx = np.random.randint(0, self.xdim - self.crop + 1)
            data_val = data_val[:,:,idx:idx+self.crop]
        else:
            # Center Crop Test Data
            data_val = data_val[:,:,self.cropidx[0]:self.cropidx[1]]
        return data_val, target


class EEGDataPreprocessor:
    def __init__(self, data_dir, valid_ratio=0.2, preprocess=True, hyperparams=None) -> None:
        self.valid_ratio = valid_ratio
        self.data = self.load(data_dir)
        self.do_preprocess = preprocess

        # hyperparameters
        if not hyperparams:
            self.params = {}
        else:
            self.params = hyperparams

        (self.x_train,
         self.y_train,
         self.x_valid,
         self.y_valid,
         self.x_test,
         self.y_test) = self.preprocess(self.data, valid_ratio)

    def get_dataset(self, data_type):
        if data_type == "train":
            return EEGDataset(self.x_train, self.y_train)
        if data_type == "valid":
            return EEGDataset(self.x_valid, self.y_valid)
        if data_type == "test":
            return EEGDataset(self.x_test, self.x_valid)
        raise ValueError("data_type must be one of 'test/train/valid'")

    def load(self, data_dir):
        X_test = np.load(f"{data_dir}/X_test.npy")
        y_test = np.load(f"{data_dir}/y_test.npy")
        X_train_valid = np.load(f"{data_dir}/X_train_valid.npy")
        y_train_valid = np.load(f"{data_dir}/y_train_valid.npy")
        person_train_valid = np.load(f"{data_dir}/person_train_valid.npy")
        person_test = np.load(f"{data_dir}/person_test.npy")
        y_train_valid -= 769
        y_test -= 769
        return (
            X_test,
            y_test,
            X_train_valid,
            y_train_valid,
            person_train_valid,
            person_test
        )

    def preprocess(self, data, valid_ratio):
        (X_test,
         y_test,
         X_train_valid,
         y_train_valid, person_train_valid, person_test) = data

        # Random splitting and reshaping the data
        # First generating the training and validation indices using random splitting

        ind_valid = np.random.choice(
            2115, int(2115*valid_ratio), replace=False)
        ind_train = np.array(list(set(range(2115)).difference(set(ind_valid))))

        # Creating the training and validation sets using the generated indices
        (x_train, x_valid) = X_train_valid[ind_train], X_train_valid[ind_valid]
        (y_train, y_valid) = y_train_valid[ind_train], y_train_valid[ind_valid]
        (person_train, person_valid) = person_train_valid[ind_train], person_train_valid[ind_valid]

        self.params['stats'] = (np.mean(x_train), np.std(x_train))

        if self.do_preprocess:
        # Preprocessing the dataset
            x_train, y_train = data_prep(x_train, y_train, person_train, self.params)
            if len(x_valid) > 0:
                x_valid, y_valid = data_prep(x_valid, y_valid, person_valid, self.params, 'test')
            X_test_prep, y_test_prep = data_prep(X_test, y_test, person_test, self.params, 'test')

            print('Shape of testing set:', X_test_prep.shape)
            print('Shape of testing labels:', y_test_prep.shape)

            print('Shape of training set:', x_train.shape)
            print('Shape of validation set:', x_valid.shape)
            print('Shape of training labels:', y_train.shape)
            print('Shape of validation labels:', y_valid.shape)
        else:
           X_test_prep, y_test_prep = X_test, y_test 

        # Converting the labels to categorical variables for multiclass classification
        y_train = to_categorical(y_train, 4)
        y_valid = to_categorical(y_valid, 4)
        y_test = to_categorical(y_test_prep, 4)
        print('Shape of training labels after categorical conversion:', y_train.shape)
        print('Shape of validation labels after categorical conversion:', y_valid.shape)
        print('Shape of test labels after categorical conversion:', y_test.shape)

        # Adding width of the segment to be 1
        x_train = x_train.reshape(
            x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_valid = x_valid.reshape(
            x_valid.shape[0], x_valid.shape[1], x_train.shape[2], 1)
        x_test = X_test_prep.reshape(
            X_test_prep.shape[0], X_test_prep.shape[1], X_test_prep.shape[2], 1)
        print('Shape of training set after adding width info:', x_train.shape)
        print('Shape of validation set after adding width info:', x_valid.shape)
        print('Shape of test set after adding width info:', x_test.shape)

        # Reshaping the training and validation dataset
        x_train = np.swapaxes(x_train, 2, 3)
        # x_train = np.swapaxes(x_train, 1,2)
        x_valid = np.swapaxes(x_valid, 2, 3)
        # x_valid = np.swapaxes(x_valid, 1,2)
        x_test = np.swapaxes(x_test, 2, 3)
        # x_test = np.swapaxes(x_test, 1,2)
        print('Shape of training set after dimension reshaping:', x_train.shape)
        print('Shape of validation set after dimension reshaping:', x_valid.shape)
        print('Shape of test set after dimension reshaping:', x_test.shape)

        return (
            x_train,
            y_train,
            x_valid,
            y_valid,
            x_test,
            y_test
        )
