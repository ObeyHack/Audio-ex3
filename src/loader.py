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

zero_to_nine = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight',
                9: 'nine'}
BATCH_SIZE = 32                                        # Batch size
TIME_STEPS = 8                                         # Input sequence length
CLASSES = 26+1                                         # Number of classes (including blank)
S_min = min([len(i) for i in zero_to_nine.values()])  # Minimum target length, for demonstration purposes,
                                                            # shortest word is 'one' with 3 letters
S_max = max([len(i) for i in zero_to_nine.values()])  # Maximum target length, for demonstration purposes,
                                                            # longest word is 'three' with 5 letters
PADDING_VALUE = 0                                      # Padding value for the input sequence
MFCC_FEATURES = 13                                     # Number of MFCC features
BLANK_LABEL = 0                                        # Blank label for CTC loss


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
    encoded_digit = torch.zeros(len(zero_to_nine[digit]))
    for i in range(len(encoded_digit)):
        char_i = zero_to_nine[digit][i]
        # encode the character
        encoded_digit[i] = ord(char_i) - 96
    return encoded_digit


def string_to_int(digit_str: str):
    """
    Convert the digit string to a number
    :param digit_str: (N, ) or (, )
    :return: int or (N, )
    """
    vals = list(zero_to_nine.values())
    if digit_str not in vals:
        return -1
    return list(zero_to_nine.values()).index(digit_str)


def decode_digit(encoded_digit: torch.Tensor):
    """
    Wrapper function for _decode_digit_not_batched,
    :param encoded_digit: shape (N, X) or (X,) where X is the length of the word (X<=T)
    :return: (str, (,)) or (N_str, (N,))
    """
    def decode_digit_not_batched(encoded_digit: torch.Tensor):
        """
        Decode the digit from the encoded digit
        :param encoded_digit: (T, )
        :return: The decoded digit as number
        """
        decoded_digit = ''
        for i in range(len(encoded_digit)):
            decoded_digit_i = int(encoded_digit[i].item())

            if decoded_digit_i == BLANK_LABEL:
                continue

            # turn class to character
            decoded_digit += chr(decoded_digit_i + 96)

        return decoded_digit


    if len(encoded_digit.shape) == 1:
        decoded = decode_digit_not_batched(encoded_digit)
        digit = torch.tensor(string_to_int(decoded))
        return decoded, digit
    else:
        decoded_batch = [decode_digit_not_batched(encoded_digit[i, :]) for i in range(encoded_digit.shape[0])]
        digits = torch.tensor([string_to_int(decoded) for decoded in decoded_batch])
        return decoded_batch, digits


def un_pad(y):
    """
    Remove the padding from the label
    :param y: Shape (N, S) or (S,)
    :return: Shape (N, X) or (X,) where X <= S
    """
    def un_pad_not_batched(y):
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

    if len(y.shape) == 1:
        return un_pad_not_batched(y).tolist()
    else:
        return [un_pad_not_batched(y[i]) for i in range(y.shape[0])]

########################################################################################################################
################################################# Data Loader ##########################################################
########################################################################################################################


def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=None)
    y_mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_FEATURES)
    return y_mfcc


def get_paths(set, root_path):
    """
    load either the train or validation or test set
    :param set: string, either 'train', 'validation' or 'test'
    :return:
    """
    # loop on every directory in the set directory (0-8) and get the path of every file in it
    paths = []
    for i in zero_to_nine:
        path_i = []
        for file in os.listdir(os.path.join(*[root_path, set, zero_to_nine[i]])):
            path_i.append(os.path.join(*[root_path, set, zero_to_nine[i], file]))
        paths.append(path_i)
    return paths


def get_set(set, root_path="data"):
    paths = get_paths(set, root_path)
    Y = []
    for i in range(len(zero_to_nine)):
        for _ in paths[i]:
            Y.append(encode_digit(i))

    X = []
    for i in range(len(zero_to_nine)):
        for file in paths[i]:
            X.append(torch.tensor(extract_mfcc(file)).T)

    X_pad = nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=PADDING_VALUE)
    Y_pad = nn.utils.rnn.pad_sequence(Y, batch_first=True, padding_value=PADDING_VALUE)
    X_pad = X_pad.permute(0, 2, 1)
    return X_pad, Y_pad


def load_data(batch_size=BATCH_SIZE):
    train_X, train_Y = get_set('train')
    validation_X, validation_Y = get_set('val')
    test_X, test_Y = get_set('test')

    # Define the datasets
    train_dataset = torch.utils.data.TensorDataset(train_X, train_Y)
    validation_dataset = torch.utils.data.TensorDataset(validation_X, validation_Y)
    test_dataset = torch.utils.data.TensorDataset(test_X, test_Y)

    # Define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=11,
                                               persistent_workers=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False,
                                                    num_workers=11, persistent_workers=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=11,
                                              persistent_workers=True)

    return {'train': train_loader, 'val': validation_loader, 'test': test_loader}


class AudioDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "data", batch_size: int = BATCH_SIZE):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self._already_called = {}
        for stage in ("fit", "validate", "test", "predict"):
            self._already_called[stage] = False

    def setup(self, stage: str) -> None:
        if self._already_called[stage]:
            return

        if stage == "fit":
            train_X, train_Y = get_set('train', self.data_dir)
            train_dataset = torch.utils.data.TensorDataset(train_X, train_Y)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                                       num_workers=11, persistent_workers=True)
            self.train_loader = train_loader
            validation_X, validation_Y = get_set('val', self.data_dir)
            validation_dataset = torch.utils.data.TensorDataset(validation_X, validation_Y)
            validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False,
                                                            num_workers=11, persistent_workers=True)
            self.val_loader = validation_loader
            self._already_called["fit"] = True
            self._already_called["validate"] = True

        if stage == "test":
            test_X, test_Y = get_set('test', self.data_dir)
            test_dataset = torch.utils.data.TensorDataset(test_X, test_Y)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                                      num_workers=11, persistent_workers=True)
            self.test_loader = test_loader
            self._already_called["test"] = True

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader



if __name__ == '__main__':
    load_data()