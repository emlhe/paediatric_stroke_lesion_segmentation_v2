import lightning as pl
import torchio as tio
import torch

class Model(pl.LightningModule):
    def __init__(self, net, criterion, learning_rate, optimizer_class, epochs):
        super().__init__()
        self.lr = learning_rate
        self.net = net
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.epochs = epochs
        # self.automatic_optimization = False

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        lr_scheduler = {
        'scheduler': torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=self.epochs),
        'name': 'lr_scheduler'
        }
        return [optimizer], [lr_scheduler]

    def prepare_batch(self, batch):
        return batch['t1']['data'], batch['seg']['data']

    def infer_batch(self, batch):
        x, y = self.prepare_batch(batch)
        # print(x.shape)
        y_hat = self.net(x)
        return y_hat, y

    def training_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)
        return loss