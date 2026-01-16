import os
import torch
from config.schema import Config
from utils.dataloader import get_dataloaders
from models.simple_cnn import SimpleCNN
from models.resnet18 import get_resnet18
import yaml

def evaluate():
    # ==================================
    # CARICA CONFIG
    # ==================================
    with open("config/config.yaml") as f:
        config = Config(**yaml.safe_load(f))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==================================
    # DATALOADER VALIDAZIONE
    # ==================================
    _, val_loader = get_dataloaders(config)

    # ==================================
    # INIZIALIZZA MODELLO
    # ==================================
    if config.model.name == "simple_cnn":
        model = SimpleCNN(num_classes=config.model.num_classes).to(device)

    elif config.model.name == "resnet18":
        model = get_resnet18(
            num_classes=config.model.num_classes,
            pretrained=config.model.pretrained
        ).to(device)

    else:
        raise ValueError(f"Modello sconosciuto: {config.model.name}")

    # ==================================
    # COSTRUISCI PERCORSO CHECKPOINT
    # ==================================
    """
    checkpoint_dir = os.path.join("checkpoints", config.model.name)
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Checkpoint non trovato: {best_model_path}")

    print(f"Caricamento pesi da: {best_model_path}")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    """
    experiment_name = config.experiment.name
    checkpoint_dir = os.path.join("experiments", experiment_name, "checkpoints")
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Checkpoint non trovato: {best_model_path}")
    
    print(f"Caricamento pesi da: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # ==================================
    # VALIDAZIONE
    # ==================================
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Accuracy finale sul validation/test: {accuracy:.4f}")