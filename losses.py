import torch
import torch.nn as nn


class BceLoss(nn.Module):
    """docstring for BCELoss"""
    def __init__(self):
        super().__init__()

    def forward(self, output):
        return nn.functional.binary_cross_entropy(
            input=output["clip_prob"],
            target=output["target"])

