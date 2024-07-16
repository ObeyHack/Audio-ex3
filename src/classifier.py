import os
import torch
import torch.nn as nn
import numpy as np
import load_data.loader as loader



class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.lstm = nn.LSTM(20, 20, 2, bidirectional=True)
        self.linear = nn.Linear(40, 9)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

