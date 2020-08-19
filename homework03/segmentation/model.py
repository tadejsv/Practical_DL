import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR

import pytorch_lightning as pl

import segmentation_models_pytorch as smp

from .loss import DiceLoss, LovaszLoss


class SegmentationModel(pl.LightningModule):
    """A U-net based segmentation model"""

    def __init__(self, backbone="resnet34", loss="bce", initial_lr=1e-3, pos_weight=6):
        super().__init__()

        self.model = smp.Unet(backbone, encoder_weights="imagenet", classes=1)
        self.initial_lr = initial_lr

        if loss == "bce":
            self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))
        elif loss == "dice":
            self.loss = DiceLoss()
        elif loss == "lovasz":
            self.loss = LovaszLoss()
        else:
            raise ValueError(f"The loss {loss} is not recognized")

        self.iou = pl.metrics.IoU(remove_bg=True)

        # Do this so experiements can be properly logged
        self.save_hyperparameters()

    def forward(self, input):
        output = self.model(input).squeeze()
        return output

    def configure_optimizers(self):
        optimizers = [optim.Adam(self.parameters(), weight_decay=0, lr=self.initial_lr)]
        schedulers = [
            {
                "scheduler": StepLR(optimizers[0], step_size=100, gamma=0.5),
                "interval": "epoch",
            }
        ]

        return optimizers, schedulers

    def training_step(self, batch, batch_idx):
        img, mask = batch
        logits = self(img)
        loss = self.loss(logits, mask)

        logs = {"train_loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        logits = self(img)
        loss = self.loss(logits, mask)

        iou = self.iou(mask, (logits > 0).int())

        return {"val_loss": loss, "val_iou": iou}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_iou = torch.stack([x["val_iou"] for x in outputs]).mean()

        logs = {"val_loss": avg_loss, "val_iou": avg_iou}
        return {"val_loss": avg_loss, "log": logs}

    def test_step(self, batch, batch_idx):
        img, mask = batch
        logits = self(img)
        loss = self.loss(logits, mask)

        iou = self.iou(mask, (logits > 0).int())

        return {"test_loss": loss, "test_iou": iou}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        test_iou = torch.stack([x["test_iou"] for x in outputs]).mean()

        logs = {"test_loss": avg_loss, "test_iou": test_iou}
        return {"test_acc": test_acc, "log": logs}
