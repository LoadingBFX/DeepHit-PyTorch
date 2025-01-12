"""
@author: bfx
@version: 1.0.0
@file: ResNet50.py
@time: 1/12/25 13:09
"""
import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()

        # Load ResNet-50 backbone
        resnet = models.resnet50(pretrained=pretrained)

        # Remove the final fully connected layer from ResNet
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove only the last FC layer


    def forward(self, x):
        feats = self.backbone(x)
        feats = torch.flatten(feats, 1)

        return feats

    def get_feature_extractor_params(self):
        return self.backbone.parameters()
