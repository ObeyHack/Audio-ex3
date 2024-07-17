import os
import librosa
import numpy as np
import torch
from librosa.feature import mfcc
from torch.utils import data

zero_to_eight = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight'}
BATCH_SIZE = 16                                        # Batch size
TIME_STEPS = 32                                        # Input sequence length
CLASSES = 26+1                                         # Number of classes (including blank)
S_min = min([len(i) for i in zero_to_eight.values()])  # Minimum target length, for demonstration purposes,
                                                            # shortest word is 'one' with 3 letters
S_max = max([len(i) for i in zero_to_eight.values()])  # Maximum target length, for demonstration purposes,
                                                            # longest word is 'three' with 5 letters
PADDING_VALUE = 0                                      # Padding value for the input sequence
MFCC_FEATURES = 20                                     # Number of MFCC features


class data_set(data.Dataset):
    def __init__(self,X,Y):
        self.X = X                           # set data
        self.Y = Y                           # set lables

    def __len__(self):
        return len(self.X)                   # return length

    def __getitem__(self, idx):
        return [self.X[idx], self.Y[idx]]


def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=None)
    y_mfcc = librosa.feature.mfcc(y=y, sr=sr)

    # pad the signal to have the same length if it is shorter than T
    if y_mfcc.shape[1] < TIME_STEPS:
        y_mfcc = np.pad(y_mfcc, ((0, 0), (0, TIME_STEPS - y_mfcc.shape[1])), 'constant')
    return y_mfcc


def get_paths(set, root_path):
    """
    load either the train or validation or test set
    :param set: string, either 'train', 'validation' or 'test'
    :return:
    """
    # loop on every directory in the set directory (0-8) and get the path of every file in it
    paths = []
    for i in zero_to_eight:
        path_i = []
        for file in os.listdir(os.path.join(*[root_path, set, zero_to_eight[i]])):
            path_i.append(os.path.join(*[root_path, set, zero_to_eight[i], file]))
        paths.append(path_i)
    return paths


def get_set(set, root_path="data"):
    paths = get_paths(set, root_path)
    Y = []
    for i in range(len(zero_to_eight)):
        for _ in paths[i]:
            Y.append(i)

    X = []
    for i in range(len(zero_to_eight)):
        for file in paths[i]:
            X.append(extract_mfcc(file))
    X = np.array(X)

    return torch.tensor(X), torch.tensor(Y)


def load_data():
    train_X, train_Y = get_set('train')
    validation_X, validation_Y = get_set('val')
    test_X, test_Y = get_set('test')

    # Define the datasets
    train_dataset = torch.utils.data.TensorDataset(train_X, train_Y)
    validation_dataset = torch.utils.data.TensorDataset(validation_X, validation_Y)
    test_dataset = torch.utils.data.TensorDataset(test_X, test_Y)

    # Define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True, )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return {'train': train_loader, 'val': validation_loader, 'test': test_loader}


if __name__ == '__main__':
    load_data()