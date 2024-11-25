import random
import timm
from enum import Enum
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights

from models.mlp import MLP
from models.spuco_model import SpuCoModel

from utils.random_seed import seed_randomness


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class SupportedModels(Enum):
    """
    Enum listing all supported models.
    """

    MLP = "mlp"
    LeNet = "lenet"
    BERT = "bert"
    DistilBERT = "distilbert"
    ResNet18 = "resnet18"
    ResNet50 = "resnet50"
    ViTB16 = "vitb16"
    ResNet50_OSE = "resnet50_ose"
    CLIP = "clip"


def model_factory(arch: str, input_shape: Tuple[int, int, int], num_classes: int, pretrained: bool = True, backbone_freeze=False,):
    """
    Factory function to create a SpuCoModel based on the specified architecture.

    :param arch: The architecture name.
    :type arch: str
    :param input_shape: The shape of the input data in the format (channels, height, width).
    :type input_shape: Tuple[int, int, int]
    :param num_classes: The number of output classes.
    :type num_classes: int
    :param pretrained: Whether to load pretrained weights. Default is True.
    :type pretrained: bool
    :return: A SpuCoModel instance.
    :rtype: SpuCoModel
    :raises NotImplementedError: If the specified architecture is not supported.
    """

    seed_randomness(random_module=random, torch_module=torch, numpy_module=np)
    arch = SupportedModels(arch)
    channel = input_shape[0]
    image_size = (input_shape[1], input_shape[2])
    backbone = None
    representation_dim = -1

    if arch == SupportedModels.MLP:
        backbone = MLP(channel * image_size[0] * image_size[1])
        representation_dim = backbone.representation_dim
    elif arch == SupportedModels.ResNet18:
        if pretrained:
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            backbone = resnet18(weights=None)
        representation_dim = backbone.fc.in_features
        backbone.fc = Identity()
    elif arch == SupportedModels.ResNet50 or arch == SupportedModels.ResNet50_OSE:
        if pretrained:
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            backbone = resnet50(weights=None)
        representation_dim = backbone.fc.in_features
        backbone.fc = Identity()
    elif arch == SupportedModels.ViTB16:
        if pretrained:
            backbone = timm.create_model("vit_base_patch16_224", pretrained=True)
        else:
            backbone = timm.create_model("vit_base_patch16_224", pretrained=False)
        representation_dim = backbone.head.in_features
        backbone.head = Identity()
    else:
        raise NotImplemented(f"Model {arch} not supported currently")
    
    print("Backbone_freeze: ", backbone_freeze)
    return SpuCoModel(backbone=backbone, representation_dim=representation_dim, num_classes=num_classes, backbone_freeze=backbone_freeze)
