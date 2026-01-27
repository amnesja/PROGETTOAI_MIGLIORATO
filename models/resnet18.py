import torch.nn as nn
from torchvision import models              #Contiene modelli pre-addestrati

def get_resnet18(num_classes=2, pretrained=True):
    # Carica modello con pesi pre-addestrati
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    """
    Se pretrained=True, il modello viene caricato con pesi
    addestrati su ImageNet (applico Transfer Learning, parto da un modello già addestrato che ha imparato a riconoscere molte caratteristiche generali delle immagini). 
    Altrimenti, viene inizializzato
    con pesi casuali.
    """

    # Sostituisci l’ultimo layer fully connected con uno adatto al tuo problema
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    """
    L'ultimo layer fully connected di ResNet18 è progettato per classificare 1000 classi (ImageNet).
    Lo sostituisco con un nuovo layer che ha il numero di output pari al numero di classi del mio problema.
    In questo modo, il modello può essere addestrato per riconoscere le classi specifiche del mio dataset.
    """

    return model
