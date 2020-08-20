# ---
# Taken from https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
# ---

import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation.lovasz_losses import lovasz_hinge


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):

        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / \
            (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class LovaszLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets):
        inputs = F.sigmoid(inputs)
        lovasz = lovasz_hinge(inputs, targets, per_image=False)
        return lovasz
