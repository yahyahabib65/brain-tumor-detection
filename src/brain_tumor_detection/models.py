from brain_tumor_detection.config import NUM_FEATURES
import torch
import torch.nn as nn
from torchvision import models


class TabularMLP(nn.Module):
  """Feed-forward neural network for the 13 extracted MRI tabular features."""

  def __init__(self, in_features: int = NUM_FEATURES):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(in_features, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(32, 1),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Squeeze last dim so output shape is (batch_size,) matching labels
    return self.net(x).squeeze(-1)


class MRICNN(nn.Module):
  """ResNet-18 based classifier for raw MRI brain scan images."""

  def __init__(self, pretrained: bool = True):
    super().__init__()
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    self.backbone = models.resnet18(weights=weights)

    in_features = self.backbone.fc.in_features
    self.backbone.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 1),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Squeeze last dim so output shape is (batch_size,) matching labels
    return self.backbone(x).squeeze(-1)