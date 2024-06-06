import torch
from torch import nn as nn
from torchvision.models import vgg16, VGG16_Weights


class VGG16FeatureExtractionNetwork(nn.Module):

    def __init__(self) -> None:
        super(VGG16FeatureExtractionNetwork, self).__init__()

        self.vgg16_features = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).features

        for parameter in self.vgg16_features.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        return self.vgg16_features(x).flatten()

    @staticmethod
    def to_tensor_input(image):
        return image
