import torch.nn as nn
from torchvision import models

def get_resnet18(num_classes=2, pretrained=True):
    # Carica modello con pesi pre-addestrati
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # Sostituisci l’ultimo layer fully connected con uno adatto al tuo problema
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model
