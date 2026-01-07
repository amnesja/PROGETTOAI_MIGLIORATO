"""
Questo modulo si occupa di:
- leggere le immagini
- applicare le trasformazioni
- creare i dataloader per l'allenamento e la validazione
"""

import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_dataloaders(config):
    """Crea e restituisce i dataloader per l'allenamento e la validazione.

    Args:
        config: oggetto Config validato (schema.py)

    Returns:
        train_loader (DataLoader): Dataloader per il set di allenamento.
        val_loader (DataLoader): Dataloader per il set di validazione.
    """

    # ==================================
    # TRASFORMAZIONI IMMAGINI
    # ==================================

    train_transforms = transforms.Compose([
        transforms.Resize((config.data.image_size, config.data.image_size)),            # Resize -> dimensione uniforme
        transforms.ToTensor(),                                                          # Converti in tensore
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])     # Normalizzazione
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((config.data.image_size, config.data.image_size)),            # Resize -> dimensione uniforme
        transforms.ToTensor(),                                                          # Converti in tensore
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])     # Normalizzazione
    ])

    # ==================================
    # DATASET
    # ==================================

    # Dataset di allenamento (training)
    train_dataset = datasets.ImageFolder(
        root=os.path.join(config.data.root_dir, "train"),
        transform=train_transforms
    )

    # Dataset di validazione (validation)
    val_dataset = datasets.ImageFolder(
        root=os.path.join(config.data.root_dir, "val"),
        transform=val_transforms
    )

    # ==================================
    # DATALOADER
    # ==================================

    # Dataloader di allenamento (training)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers
    )

    #dataloader di validazione (validation)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers
    )

    return train_loader, val_loader