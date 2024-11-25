import random

import numpy as np
import torch
from torch import nn

from utils.random_seed import seed_randomness


class SpuCoModel(nn.Module):
    """
    Wrapper module to allow for methods that use penultimate layer embeddings
    to easily access this via *backbone*
    """

    def __init__(self, backbone: nn.Module, representation_dim: int, num_classes: int, backbone_freeze: bool = False):
        """
        Initializes a SpuCoModel

        :param backbone: The backbone network.
        :type backbone: torch.nn.Module
        :param representation_dim: The dimensionality of the penultimate layer embeddings.
        :type representation_dim: int
        :param num_classes: The number of output classes.
        :type num_classes: int
        """

        seed_randomness(random_module=random, torch_module=torch, numpy_module=np)
        super().__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        self.classifier = nn.Linear(representation_dim, num_classes)
        if backbone_freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
    def forward(self, x, modality='image'):
        """
        Forward pass of the SpuCoModel.

        :param x: Input tensor.
        :type x: torch.Tensor
        :return: Output tensor.
        :rtype: torch.Tensor
        """
        return self.classifier(self.backbone(x).float())
