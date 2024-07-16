import torch
import torch.nn as nn
import loader
import lightning as L



class NeuralNetwork(L.LightningModule):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.lstm = nn.LSTM(20, 20, 2, bidirectional=True)
        self.linear = nn.Linear(40, 9)
        self.lr = 0.001

    def forward(self, x):
        x, _ = self.lstm(x)
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