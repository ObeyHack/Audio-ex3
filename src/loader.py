import os
import librosa
import numpy as np
import torch
from librosa.feature import mfcc
from torch import nn
from torch.utils import data

zero_to_eight = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight'}
BATCH_SIZE = 16                                        # Batch size
TIME_STEPS = 32                                        # Input sequence length
CLASSES = 26+2                                         # Number of classes (including blank and end of sequence)
S_min = min([len(i) for i in zero_to_eight.values()])  # Minimum target length, for demonstration purposes,
                                                            # shortest word is 'one' with 3 letters
S_max = max([len(i) for i in zero_to_eight.values()])  # Maximum target length, for demonstration purposes,
                                                            # longest word is 'three' with 5 letters
PADDING_VALUE = 0                                      # Padding value for the input sequence
MFCC_FEATURES = 20                                     # Number of MFCC features



def encode_digit(digit: int):
    """
    Encode the digit i into a tensor base on digit word size,
    Ascii(a) = 97, Ascii(z) = 122
    :param digit: The digit to encode
    :return: The tensor of size CLASSES
    """
    encoded_digit = torch.zeros(len(zero_to_eight[digit]))
    for i in range(len(encoded_digit)):
        char_i = zero_to_eight[digit][i]
        # encode the character
        encoded_digit[i] = ord(char_i) - 96
    return encoded_digit


def _decode_digit_not_batched(encoded_digit: torch.Tensor):
    """
    pick the most probable class for each time step and decode the digit from the classes, if it
    a blank size continue to the next time step
    :param encoded_digit: matrix of TxC shape, one for each time step a probability distribution
                        over the classes
    :return: The decoded digit as number
    """
    decoded_digit = ''
    for i in range(len(encoded_digit)):
        probs = encoded_digit[i]
        # pick the most probable class
        decoded_digit_i = torch.argmax(probs).item()
        # if the class is the blank class continue to the next time step
        if decoded_digit_i == CLASSES:
            continue

        # turn class to character
        decoded_digit += chr(decoded_digit_i + 96)

    # turn digit string to number
    vals = list(zero_to_eight.values())
    if decoded_digit not in vals:
        return -1
    digit = list(zero_to_eight.values()).index(decoded_digit)
    return digit


def decode_digit(encoded_digit: torch.Tensor):
    """
    Wrapper function for _decode_digit_not_batched,
    :param encoded_digit: shape (T, C) or (T, N, C)
    :return: (,) or (N,) tensor with the decoded digits
    """
    if len(encoded_digit.shape) == 2:
        return _decode_digit_not_batched(encoded_digit)
    else:
        return torch.tensor([_decode_digit_not_batched(encoded_digit[:, i, :]) for i in range(encoded_digit.shape[1])])



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
            Y.append(encode_digit(i))

    X = []
    for i in range(len(zero_to_eight)):
        for file in paths[i]:
            X.append(extract_mfcc(file))
    X = np.array(X)

    Y_pad = nn.utils.rnn.pad_sequence(Y, batch_first=True, padding_value=CLASSES)
    return torch.tensor(X), Y_pad


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
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                                    num_workers=11, persistent_workers=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=11)

    return {'train': train_loader, 'val': validation_loader, 'test': test_loader}


if __name__ == '__main__':
    load_data()