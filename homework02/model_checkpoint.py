import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.checkpoint import checkpoint_sequential

import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy

from model import BasicModule


class MyNetCheckpoint(pl.LightningModule):
    """Baby ResNet model (for gradient checkpointing)
    
    This version includes 4 residual layers and 2 fully connected layers.
    
    Input: 3*64*64 image
    Layer 0: 5*5 convolution with 16 channels and stride 2
    Layer 1: 4 residual blocks of 2 3*3 convolutions with 32 channels
    Layer 2: 4 residual blocks of 2 3*3 convolutions with 64 channels
    Layer 3: 4 residual blocks of 2 3*3 convolutions with 128 channels
    
    FC1: Layer with 500 neurons (and ReLU activation)
    FC2: Layer with 200 neurons
    """

    def __init__(
        self,
        steps_per_epoch,
        warmup_epochs=10,
        decay_epochs=50,
        initial_channel_size=32,
        checkpoint_stages=2,
    ):
        super(MyNetCheckpoint, self).__init__()

        # Some important variables
        self.WARMUP_EPOCHS = warmup_epochs
        self.DECAY_EPOCHS = decay_epochs
        self.steps_per_epoch = steps_per_epoch
        self.checkpoint_stages = checkpoint_stages

        initial_channel = initial_channel_size

        # Metric
        self.metric = Accuracy()

        # Layers of the model
        self.layer_0 = nn.Conv2d(3, initial_channel, 5, padding=2, stride=1)

        self.features = nn.Sequential()
        self._make_layer(
            3, initial_channel * 1, initial_channel * 2, 1
        )  # Returns 32 * 32
        self._make_layer(
            6, initial_channel * 2, initial_channel * 4, 2
        )  # Returns 16 * 16
        self._make_layer(
            6, initial_channel * 4, initial_channel * 8, 3
        )  # Returns 8 * 8
        self.features.add_module("avg_pool", nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = nn.Linear(initial_channel * 8, 200)

    def _make_layer(self, n_blocks, in_ch, out_ch, stage, downsample=True):

        self.features.add_module(
            f"block_{stage}_0", BasicModule(in_ch, out_ch, downsample=downsample)
        )
        for i in range(n_blocks - 1):
            self.features.add_module(
                f"block_{stage}_{i+1}", BasicModule(out_ch, out_ch)
            )

    def forward(self, input):
        out = self.layer_0(input)
        out = checkpoint_sequential(self.features, self.checkpoint_stages, out)

        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)

        return out

    def configure_optimizers(self):
        total_epochs = self.WARMUP_EPOCHS + self.DECAY_EPOCHS

        optimizers = [optim.Adam(self.parameters(), weight_decay=1e-4)]
        schedulers = [
            {
                "scheduler": OneCycleLR(
                    optimizers[0],
                    epochs=total_epochs,
                    steps_per_epoch=self.steps_per_epoch,
                    pct_start=self.WARMUP_EPOCHS / total_epochs,
                    final_div_factor=1e3,
                    div_factor=1e2,
                    max_lr=1e-2,
                ),
                "interval": "step",
            }
        ]

        return optimizers, schedulers

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        labels_hat = torch.argmax(logits, dim=1)

        accuracy = self.metric(y, labels_hat)

        return {"val_loss": loss, "val_acc": accuracy}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()

        tensorboard_logs = {"val_loss": avg_loss, "val_acc": avg_acc}
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        labels_hat = torch.argmax(logits, dim=1)

        accuracy = self.metric(y, labels_hat)

        return {"test_loss": loss, "test_acc": accuracy}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        test_acc = torch.stack([x["test_acc"] for x in outputs]).mean()

        tensorboard_logs = {"test_loss": avg_loss, "test_acc": test_acc}
        return {"test_acc": test_acc, "log": tensorboard_logs}
