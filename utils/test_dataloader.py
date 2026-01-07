import yaml
from config.schema import Config
from utils.dataloader import get_dataloaders

# Caricamento file di configurazione di test
with open("config/config.yaml", "r") as f:
    config = Config(**yaml.safe_load(f))

# Creazione dei dataloader di test
train_loader, val_loader = get_dataloaders(config)

#Prendiamo un batch di dati di esempio
images, labels = next(iter(train_loader))

print("Shape delle immagini:", images.shape)
print("Shape delle etichette:", labels.shape)