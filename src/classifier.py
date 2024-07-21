import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import loader
import lightning as L
from neptune.types import File
from io import StringIO

class NeuralNetwork(L.LightningModule):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Input size:  MFCC_FEATURESxT where MFCC_FEATURES is the number of MFCC features and T is the # of time steps
        # Output size: TxC where T is the number of time steps and C is the number of classes
        self.layers_count = 100
        self.kernel_filter = 1000
        self.lstm = torch.nn.LSTM(input_size=loader.MFCC_FEATURES, hidden_size=loader.CLASSES,
                                  num_layers=self.layers_count, batch_first=True)
        self.cnv = nn.Conv2d(in_channels=1, out_channels=self.kernel_filter, kernel_size=(3, 3), padding=1)

        # same as weighted sum of the input
        self.conv1 = nn.Conv2d(in_channels=self.kernel_filter, out_channels=1, kernel_size=(1, 1), padding=0, stride=1)
        self.relu = nn.ReLU()
        self.loss = nn.CTCLoss()
        self.lr = 0.1

    def forward(self, x):
        """
        :param x: (N, T, MFCC_FEATURES) where N is the batch size, T is the number of time steps and MFCC_FEATURES is the
        :return:
        """
        x = x.permute(0, 2, 1)

        # (N, MFCC_FEATURES, T)
        res = self.lstm(x)
        x = res[0]
        x = self.relu(x)

        # (N, T, C)
        x = x[:, None, :, :]

        # (N, 1, T, C)
        x = self.cnv(x)

        # (N, self.kernel_filter, T, C)
        x = self.conv1(x)

        # (N, 1, T, C)
        x = x.squeeze(1)

        # (N, T, C)
        x = nn.functional.log_softmax(x, dim=2)
        x = x.permute(1, 0, 2)

        # (T, N, C)
        return x


    def CTCLoss(self, y_hat, y):
        """
        :param y_hat: The predicted values, shape (T, N, C) where T is TimeSteps, N is the batch size and
                        C is the number of classes.
                        In total, they are N matrices of TxC shape, one for each time step a probability distribution
                        over the classes
        :param y: The true values, shape (N, S) where N is the batch size and S is the length of the word.
        :return: The CTC loss.
        """
        batch_size = y_hat.shape[1]
        # label_length is the length of the text label. In our case is the length of the word
        # find where the padding starts
        un_padded_y = loader.un_pad(y)
        label_length = torch.tensor([len(label) for label in un_padded_y]) * torch.ones(batch_size, dtype=torch.long)

        # The input length is number of time steps
        input_lengths = torch.full(size=(batch_size,), fill_value=loader.TIME_STEPS, dtype=torch.long)
        return self.loss(y_hat, y, input_lengths, label_length)

    def predict(self, y_hat):
        """
        pick the most probable class for the prediction
        :param y_hat: shape (T, C) or (T, N, C)
        :return: int or (N, ) where N is the batch size
        """

        def un_batched_predict(y_hat):
            """
            pick the most probable class for the prediction
            :param y_hat: shape (T, C)
            :return: string
            """
            vals = loader.zero_to_eight.values()
            loss = {vals: 0 for vals in vals}
            for i, val in enumerate(vals):
                encoded_digit = loader.encode_digit(i)
                loss[val] = self.loss(y_hat,encoded_digit, y_hat.shape[:1], encoded_digit.shape[:1])

            # return the class with the lowest loss
            return min(loss, key=loss.get)

        if len(y_hat.shape) == 2:
            return un_batched_predict(y_hat)

        else:
            return [un_batched_predict(y_hat[:, i, :]) for i in range(y_hat.shape[1])]


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # calculate the loss
        loss = self.CTCLoss(y_hat, y)

        # log the loss
        self.log_dict({'train_loss': loss.item()})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.CTCLoss(y_hat, y)

        decoded_y, digits = loader.decode_digit(y)
        digits_hat_str = self.predict(y_hat)
        digits_hat = torch.tensor([loader.string_to_int(digit) for digit in digits_hat_str])

        # log the decoded values
        df = pd.DataFrame({'y': decoded_y, 'y_hat': digits_hat_str})

        # log the accuracy
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        self.logger.experiment[f"training/val_predictions_{self.current_epoch}"].upload(File.from_stream(csv_buffer, extension="csv"))

        acc = torch.sum(torch.eq(digits, digits_hat)) / len(digits)
        self.log_dict({'val_loss': loss.item(), 'val_acc': acc.item()})

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.CTCLoss(y_hat, y)

        decoded_y, digits = loader.decode_digit(y)
        digits_hat_str = self.predict(y_hat)
        digits_hat = torch.tensor([loader.string_to_int(digit) for digit in digits_hat_str])

        # log the decoded values
        df = pd.DataFrame({'y': decoded_y, 'y_hat': digits_hat_str})

        # log the accuracy
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        self.logger.experiment[f"training/test_predictions_{self.current_epoch}"].upload(
            File.from_stream(csv_buffer, extension="csv"))

        acc = torch.sum(torch.eq(digits, digits_hat)) / len(digits)
        self.log_dict({'test_loss': loss.item(), 'test_acc': acc.item()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


def main():
    data_loader = loader.load_data()
    model = NeuralNetwork()
    trainer = L.Trainer(accelerator="auto", devices="auto", strategy="auto")
    #trainer.fit(model, data_loader['train'])
    trainer.fit(model, data_loader['train'], data_loader['val'])
    trainer.test(model, data_loader['test'])


if __name__ == '__main__':
    main()