import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_lightning.loggers import NeptuneLogger
import loader
import lightning as L
from neptune.types import File
from io import StringIO


default_config = {
    "lr": 1e-3,
    "n_hidden": 512,
    "dropout": 0.1,
    "batch_size": loader.BATCH_SIZE,
}


class DigitClassifier(L.LightningModule):
    def __init__(self, n_feature, config: dict, n_class=loader.CLASSES):
        super(DigitClassifier, self).__init__()
        self.n_feature = n_feature
        self.n_class = n_class
        self.in_channels = 1
        self.out_channels = config['n_hidden'] // n_feature
        self.n_hidden = self.out_channels * n_feature
        self.lr = config['lr']
        self.dropout = config['dropout']

        self.save_hyperparameters(config)

        # Evaluation metrics
        self.eval_loss = []
        self.eval_accuracy = []
        self.test_loss = []
        self.test_accuracy = []

        # Non-layer modules
        self.batch_norm = nn.BatchNorm1d(self.n_hidden, affine=False)
        self.loss = nn.CTCLoss()

        # Input size:  FxT where F is the number of MFCC features and T is the # of time steps
        # Output size: TxC where T is the number of time steps and C is the number of classes
        self.conv = nn.Sequential(
                        nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                  kernel_size=((self.n_class//2)*2+1, 7), padding=(self.n_class//2, 3), bias=True),
                        nn.BatchNorm2d(self.out_channels, affine=False),
                        torch.nn.Hardtanh())

        self.bi_rnn = torch.nn.LSTM(self.n_hidden, self.n_hidden, num_layers=1,
                            dropout=self.dropout, batch_first=True, bias=True, bidirectional=True)

        self.linear_final = nn.Linear(in_features=self.n_hidden*2, out_features=self.n_class)

    def forward(self, x):
        """
        :param x: (N, F, T) where N is the batch size, T is the number of time steps and F is the number of features
        :return:
        """
        # (N, F, T)
        x = x.unsqueeze(1)

        # (N, C_in, F, T)
        x = x.permute(0, 1, 3, 2)

        # (N, C_in, T, F)
        x = self.conv(x)

        # (N, C_out, T, F)
        x = x.permute(0, 1, 3, 2)

        # (N, C_out, F, T)
        x = x.reshape(x.size(0), x.size(1) * x.size(2), x.size(3))

        # (N, C_out * F,  T)
        x = self.batch_norm(x)

        # (N, T, C_out * F)
        x = x.permute(0, 2, 1)

        # (N, T, C_out * F)
        x, _ = self.bi_rnn(x)

        # (N, T, H)
        x = self.linear_final(x)

        # (N, T, C)
        x = nn.functional.log_softmax(x, dim=2)
        x = x.permute(1, 0, 2)

        # (T, N, C)
        return x

    def ctc_loss(self, y_hat, y):
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
        input_lengths = torch.full(size=(batch_size,), fill_value=y_hat.shape[0], dtype=torch.long)
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
            vals = loader.zero_to_nine.values()
            loss = {vals: math.inf for vals in vals}
            for i, val in enumerate(vals):
                encoded_digit = loader.encode_digit(i)
                loss[val] = self.loss(y_hat,encoded_digit, y_hat.shape[:1], encoded_digit.shape[:1])

            # return the class with the lowest loss
            return min(loss, key=loss.get)

        if len(y_hat.shape) == 2:
            return un_batched_predict(y_hat)

        else:
            return [un_batched_predict(y_hat[:, i, :]) for i in range(y_hat.shape[1])]

    def accuracy(self, y_hat, y):
        """
        :param y_hat: The predicted values, shape (T, N, C) where T is TimeSteps, N is the batch size and
                        C is the number of classes.
                        In total, they are N matrices of TxC shape, one for each time step a probability distribution
                        over the classes
        :param y: The true values, shape (N, S) where N is the batch size and S is the length of the word.
        :return: The accuracy.
        """
        decoded_y, digits = loader.decode_digit(y)
        digits_hat_str = self.predict(y_hat)
        digits_hat = torch.tensor([loader.string_to_int(digit) for digit in digits_hat_str])

        acc = torch.sum(torch.eq(digits, digits_hat)) / len(digits)
        return acc

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # calculate the loss
        loss = self.ctc_loss(y_hat, y)

        # log the loss
        self.log_dict({'train_loss': loss.item()})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.ctc_loss(y_hat, y)
        acc = self.accuracy(y_hat, y)

        self.eval_loss.append(loss)
        self.eval_accuracy.append(acc)
        return {"val_loss": loss, "val_accuracy": acc}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.ctc_loss(y_hat, y)
        acc = self.accuracy(y_hat, y)

        self.test_loss.append(loss)
        self.test_accuracy.append(acc)
        return {"test_loss": loss, "test_accuracy": acc}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.eval_loss).mean()
        avg_acc = torch.stack(self.eval_accuracy).mean()
        self.log("val_avg_loss", avg_loss)
        self.log("val_avg_accuracy", avg_acc)
        self.eval_loss.clear()
        self.eval_accuracy.clear()

    def on_test_epoch_end(self):
        avg_loss = torch.stack(self.test_loss).mean()
        avg_acc = torch.stack(self.test_accuracy).mean()
        self.log("val_avg_loss", avg_loss)
        self.log("val_avg_accuracy", avg_acc)
        self.test_loss.clear()
        self.test_accuracy.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def train_func(config=None, dm=None, model=None, logger=None, logger_config=None, num_epochs=10):
    if config is None:
        config = default_config
    if dm is None:
        dm = loader.AudioDataModule(batch_size=config['batch_size'])
    if logger is None and logger_config is not None:
        logger = NeptuneLogger(**logger_config)

    if model is None:
        n_feature = dm.train_loader.dataset.tensors[0].shape[1]
        model = DigitClassifier(n_feature, config)

    # log the hyperparameters and not the api key and project name
    logger.run["parameters"] = config

    trainer = L.Trainer(
        devices="auto",
        accelerator="auto",
        logger=logger,
        max_epochs=num_epochs,
    )
    trainer.fit(model, datamodule=dm)

    logger.run.stop()
    return trainer


def main():
    train_func()


if __name__ == '__main__':
    main()