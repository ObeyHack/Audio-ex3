import os
import librosa
import numpy as np
import torch
from librosa.feature import mfcc
from torch import nn
from torch.utils import data
import lightning as L

########################################################################################################################
################################################# Constants ############################################################
########################################################################################################################

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
BLANK_LABEL = 26                                       # Blank label for CTC loss


########################################################################################################################
################################################# Encode/Decode ########################################################
########################################################################################################################

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
    Decode the digit from the encoded digit
    :param encoded_digit: (T, )
    :return: The decoded digit as number
    """
    decoded_digit = ''
    for i in range(len(encoded_digit)):
        decoded_digit_i = encoded_digit[i].item()

        if decoded_digit_i == BLANK_LABEL:
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
    :param encoded_digit: shape (N, X) or (X,) where X is the length of the word (X<=T)
    :return: (,) or (N,) tensor with the decoded digits
    """
    if len(encoded_digit.shape) == 1:
        return _decode_digit_not_batched(encoded_digit)
    else:
        return torch.tensor([_decode_digit_not_batched(encoded_digit[i, :]) for i in range(encoded_digit.shape[1])])


def _un_pad_not_batched(y):
    """
    Remove the padding from the label
    :param y: Shape (S,)
    :return: Shape (X,) where X <= S
    """
    un_pad_i = -1
    for i in range(len(y), 0, -1):
        if y[i - 1] != PADDING_VALUE:
            un_pad_i = i
            break

    return y[:un_pad_i]

def un_pad(y):
    """
    Remove the padding from the label
    :param y: Shape (N, S) or (S,)
    :return: Shape (N, X) or (X,) where X <= S
    """
    if len(y.shape) == 1:
        return _un_pad_not_batched(y).tolist()
    else:
        return [_un_pad_not_batched(y[i]) for i in range(y.shape[0])]

########################################################################################################################
################################################# Data Loader ##########################################################
########################################################################################################################


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

    Y_pad = nn.utils.rnn.pad_sequence(Y, batch_first=True, padding_value=PADDING_VALUE)
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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=11)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                                    num_workers=11, persistent_workers=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=11,
                                              persistent_workers=True)

    return {'train': train_loader, 'val': validation_loader, 'test': test_loader}


class AudioDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "data"):
        super().__init__()
        self.data_dir = data_dir

    def setup(self, stage: str) -> None:
        if stage == "fit":
            train_X, train_Y = get_set('train', self.data_dir)
            train_dataset = torch.utils.data.TensorDataset(train_X, train_Y)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            self.train_loader = train_loader

        if stage == "validate" or stage == "fit":
            validation_X, validation_Y = get_set('val', self.data_dir)
            validation_dataset = torch.utils.data.TensorDataset(validation_X, validation_Y)
            validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
            self.val_loader = validation_loader

        if stage == "test":
            test_X, test_Y = get_set('test', self.data_dir)
            test_dataset = torch.utils.data.TensorDataset(test_X, test_Y)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
            self.test_loader = test_loader

        print(stage)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader



if __name__ == '__main__':
    load_data()