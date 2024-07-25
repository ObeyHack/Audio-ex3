import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightning.pytorch.loggers import NeptuneLogger
import loader
import lightning as L
from neptune.types import File
from io import StringIO


default_config = {
    "lr": 1e-3,
    "layers_count": 1,
    "hidden_size": 2048,
    "batch_size": 32,
}


class DigitClassifier(L.LightningModule):
    def __init__(self, config: dict):
        super(DigitClassifier, self).__init__()
        self.layers_count = config['layers_count']
        self.hidden_size = config['hidden_size']
        self.lr = config['lr']
        self.save_hyperparameters(config)

        self.eval_loss = []
        self.eval_accuracy = []

        # Input size:  MFCC_FEATURESxT where MFCC_FEATURES is the number of MFCC features and T is the # of time steps
        # Output size: TxC where T is the number of time steps and C is the number of classes
        self.lstm = torch.nn.LSTM(input_size=loader.MFCC_FEATURES, hidden_size=self.hidden_size,
                                  num_layers=self.layers_count, batch_first=True)

        self.linear = nn.Linear(in_features=self.hidden_size, out_features=loader.CLASSES)

        # same as weighted sum of the input
        self.relu = nn.ReLU()
        self.loss = nn.CTCLoss()

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
            vals = loader.zero_to_nine.values()
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

    def forward(self, x):
        """
        :param x: (N, T, MFCC_FEATURES) where N is the batch size, T is the number of time steps and MFCC_FEATURES is the
        :return:
        """
        # (N, T, MFCC_FEATURES)
        x = x.permute(0, 2, 1)

        # (N, MFCC_FEATURES, T)
        x, _ = self.lstm(x)
        x = self.relu(x)

        # (N, T, C)
        x = self.linear(x)

        # (N, T, C)
        x = nn.functional.log_softmax(x, dim=2)
        x = x.permute(1, 0, 2)

        # (T, N, C)
        return x

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
        return {"val_loss": loss, "val_accuracy": acc}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.eval_loss).mean()
        avg_acc = torch.stack(self.eval_accuracy).mean()
        self.log("avg_loss", avg_loss, sync_dist=True)
        self.log("avg_accuracy", avg_acc, sync_dist=True)
        self.eval_loss.clear()
        self.eval_accuracy.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def train_func(config=None, dm=None, model=None, logger=None, logger_config=None, num_epochs=10):
    if config is None:
        config = default_config
    if dm is None:
        dm = loader.AudioDataModule(batch_size=config['batch_size'])
    if model is None:
        model = DigitClassifier(config)
    if logger is None:
        logger = NeptuneLogger(project=logger_config["project_name"], api_key=logger_config["api_key"],
                               log_model_checkpoints=False)

    # log the hyperparameters and not the api key and project name
    logger.run["parameters"] = config

    metrics = {"loss": "val_loss", "acc": "val_accuracy"}
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