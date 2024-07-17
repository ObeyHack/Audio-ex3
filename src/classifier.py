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
    return (((W - K + 2*P)/S) + 1)


class NeuralNetwork(L.LightningModule):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.cnv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), padding=1)
        self.linear = nn.Linear(40, 9)
        self.softmax = nn.Softmax()
        self.lr = 0.001

    def forward(self, x):
        x = self.cnv(x)
        x = self.linear(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CTCLoss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CTCLoss(y_hat, y)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CTCLoss(y_hat, y)
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