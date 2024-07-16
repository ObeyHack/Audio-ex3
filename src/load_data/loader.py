import os
import librosa
import numpy as np
from librosa.feature import mfcc

zero_to_eight = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight'}

def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return mfcc(y=y, sr=sr)


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


def get_set(set, root_path="../data"):
    paths = get_paths(set, root_path)
    Y = []
    for i in range(len(zero_to_eight)):
        for _ in paths[i]:
            Y.append(i)

    X = []
    for i in range(len(zero_to_eight)):
        for file in paths[i]:
            X.append(extract_mfcc(file))

    return X, Y


def load_data():
    train_X, train_Y = get_set('train')
    validation_X, validation_Y = get_set('validation')
    test_X, test_Y = get_set('test')
    return {'train': (train_X, train_Y), 'validation': (validation_X, validation_Y), 'test': (test_X, test_Y)}


if __name__ == '__main__':
    load_data()