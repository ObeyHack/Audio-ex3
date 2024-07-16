import os
import torch
import torch.nn as nn
import numpy as np
import load_data.loader as loader


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.lstm = nn.LSTM(20, 20, 2, bidirectional=True)
        self.linear = nn.Linear(40, 9)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x



def main():
    model = NeuralNetwork().to(device)
    data_loaders = loader.load_data()
    loss = nn.CTCLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


if __name__ == '__main__':
    main()