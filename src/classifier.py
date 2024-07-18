import torch
import torch.nn as nn
import loader
import lightning as L


class NeuralNetwork(L.LightningModule):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Input size:  MFCC_FEATURESxT where MFCC_FEATURES is the number of MFCC features and T is the # of time steps
        # Output size: TxC where T is the number of time steps and C is the number of classes
        self.lstm = torch.nn.LSTM(input_size=loader.MFCC_FEATURES, hidden_size=loader.CLASSES, batch_first=True)
        self.cnv = nn.Conv2d(in_channels=1, out_channels=200, kernel_size=(3, 3), padding=1)
        self.relu = nn.ReLU()
        self.loss = nn.CTCLoss()
        self.lr = 0.001

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.lstm(x)
        x = x[0]
        x = x[:, None, :, :]
        x = self.cnv(x)
        # squeeze the output
        x = x.squeeze(1)
        nn.functional.log_softmax(x, dim=3)
        x = x.permute(1,2, 0, 3)
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


    def argmax_prob(self, y_hat):
        """
        pick the most probable class for each time step and decode the digit from the classes
        :param y_hat: shape (T, C) or (T, N, C)
        :return: (,) or (N,) tensor with the decoded digits
        """
        if len(y_hat.shape) == 2:
            return torch.argmax(y_hat, dim=1)
        else:
            return torch.stack([torch.argmax(y_hat[:, i, :], dim=1) for i in range(y_hat.shape[1])])


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # calculate the loss
        #now y_hat is of shape (number_of_kernels,T, N, C), so we need to run over the number of kernels and sum the loss
        loss=0
        for i in range(y_hat.shape[0]):
            loss = loss + self.CTCLoss(y_hat[i], y)
        # log the loss
        self.log_dict({'train_loss': loss.item()})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = 0
        for i in range(y_hat.shape[0]):
            loss = loss + self.CTCLoss(y_hat[i], y)
        #loss = self.CTCLoss(y_hat, y)
        acc=0
        for i in range(y_hat.shape[0]):
            argmax_y_hat = self.argmax_prob(y_hat[i])
            digits = loader.decode_digit(y)
            digits_hat = loader.decode_digit(argmax_y_hat)

            acc =acc+ torch.sum(torch.eq(digits, digits_hat)) / len(digits)
        self.log_dict({'val_loss': loss.item(), 'val_acc': acc.item()})

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = 0
        for i in range(y_hat.shape[0]):
            loss = loss + self.CTCLoss(y_hat[i], y)
        # loss = self.CTCLoss(y_hat, y)
        acc = 0
        for i in range(y_hat.shape[0]):
            argmax_y_hat = self.argmax_prob(y_hat[i])
            digits = loader.decode_digit(y)
            digits_hat = loader.decode_digit(argmax_y_hat)

            acc = acc + torch.sum(torch.eq(digits, digits_hat)) / len(digits)
        self.log_dict({'test_loss': loss.item(), 'test_acc': acc.item()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def main():
    data_loader = loader.load_data()
    model = NeuralNetwork()
    trainer = L.Trainer(accelerator="auto", devices="auto", strategy="auto")
    #trainer.fit(model, data_loader['train'])
    trainer.fit(model, data_loader['train'], data_loader['val'])
    trainer.test(model, data_loader['test'])


if __name__ == '__main__':
    main()
