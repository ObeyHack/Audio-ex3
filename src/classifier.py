import torch
import torch.nn as nn
import loader
import lightning as L


def cnn_output_shape(W, K, P, S):
    """
    (((W - K + 2P)/S) + 1)
    :param W: Input size
    :param K: Filter size
    :param P: Stride
    :param S: Padding
    :return: The output size
    """
    return (((W - K + 2 * P) / S) + 1)


class NeuralNetwork(L.LightningModule):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Input size:  MFCC_FEATURESxT where MFCC_FEATURES is the number of MFCC features and T is the number of time steps
        # Output size: TxC where T is the number of time steps and C is the number of classes

        self.lstm = torch.nn.LSTM(input_size=loader.MFCC_FEATURES, hidden_size=10, num_layers=1, batch_first=True)

        self.cnv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), padding=1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.lr = 0.001

    def forward(self, x):
        x = self.lstm(x)



        x = x[:, None, :, :]
        x = self.cnv(x)
        # add pooling
        x = self.max_pooling(x)
        # vectorize the output
        x = x.view(x.size(0), -1)
        return x


    def _CTCLoss(self, y_hat, y):
        """
        :param y_hat: The predicted values, shape (T, N, C) where T is TimeSteps, N is the batch size and
                        C is the number of classes.
                        In total, they are N matrices of TxC shape, one for each time step a probability distribution
                        over the classes
        :param y: The true values, shape (N, S) where N is the batch size and S is the length of the word.
        :return: The CTC loss.
        """
        batch_size = y_hat.shape[1]

        # The input length is number of time steps
        input_lengths = torch.full(size=(batch_size,), fill_value=loader.TIME_STEPS, dtype=torch.long)

        # label_length is the length of the text label. In our case is the length of the word
        # label_length = torch.tensor([len(loader.zero_to_eight[y_i]) for y_i in y])
        label_length = torch.tensor([len(loader.zero_to_eight[y_i]) for y_i in y])
        return nn.CTCLoss(y_hat, y, input_lengths, label_length)


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        # calculate the loss
        loss = self._CTCLoss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CTCLoss(y_hat, y)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self._CTCLoss(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def main():
    data_loader = loader.load_data()
    model = NeuralNetwork()
    trainer = L.Trainer(accelerator="auto", devices="auto", strategy="auto")
    trainer.fit(model, data_loader['train'], data_loader['val'])
    trainer.test(model, data_loader['test'])


if __name__ == '__main__':
    main()
