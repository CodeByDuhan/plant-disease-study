import torch.nn as nn
from torchvision import models


def build_resnet18_classifier(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    ResNet18 for multi-class classification.
    Loads ImageNet pretrained weights and replaces the final fc
    layer to match num_classes.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    m = models.resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def freeze_all_backbone(model: nn.Module) -> None:
    """Freeze everything except the classification head (fc)."""
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("fc.")