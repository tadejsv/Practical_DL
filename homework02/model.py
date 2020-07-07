import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy


class BasicModule(nn.Module):
    """Basic 2 layer 3x3 convnet block
    
    Contains 2 3*3 convolution layers. If downsampling, the first convolution layer has a stride of 2,
    and the input is passed through a 1*1 convolution layer with stride 2 before adding at the end.
    """

    def __init__(self, in_ch, out_ch, downsample=False):
        super(BasicModule, self).__init__()

        if downsample:
            stride = 2
            self.downsample = nn.Conv2d(in_ch, out_ch, 1, stride=2)
        elif in_ch != out_ch:
            stride = 1
            self.downsample = nn.Conv2d(in_ch, out_ch, 1, stride=1)
        else:
            stride = 1
            self.downsample = nn.Identity()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):

        out = self.conv1(input)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.downsample(input)
        out = self.relu(out)

        return out


class MyNet(pl.LightningModule):
    """Baby ResNet model
    
    This version includes 3 residual layers and 1 fully connected layer.
    
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
    ):
        super(MyNet, self).__init__()

        # Some important variables
        self.WARMUP_EPOCHS = warmup_epochs
        self.DECAY_EPOCHS = decay_epochs
        self.steps_per_epoch = steps_per_epoch

        initial_channel = initial_channel_size

        # Metric
        self.metric = Accuracy()

        # Layers of the model
        self.layer_0 = nn.Conv2d(3, initial_channel, 5, padding=2, stride=1)
        self.layer_1 = self._make_layer(
            3, initial_channel, initial_channel * 2
        )  # Returns 32 * 32
        self.layer_2 = self._make_layer(
            6, initial_channel * 2, initial_channel * 4
        )  # Returns 16 * 16
        self.layer_3 = self._make_layer(
            6, initial_channel * 4, initial_channel * 8
        )  # Returns 8 * 8
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(initial_channel * 8, 200)

    def _make_layer(self, n_blocks, in_ch, out_ch, downsample=True):

        blocks = [BasicModule(in_ch, out_ch, downsample=downsample)]
        for i in range(n_blocks - 1):
            blocks.append(BasicModule(out_ch, out_ch))

        return nn.Sequential(*blocks)

    def forward(self, input):

        out = self.layer_0(input)
        out = self.layer_1(out)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.avg_pool(out)

        out = torch.flatten(out, start_dim=1)
        out = self.fc1(out)

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


# Initialize all weights (He initialization)
def init_fn(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)