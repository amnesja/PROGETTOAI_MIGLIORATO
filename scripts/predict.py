import os
import torch
import yaml
from PIL import Image
from torchvision import transforms

from config.schema import Config
from models.simple_cnn import SimpleCNN
from models.resnet18 import get_resnet18


def load_model(config, device):
    """Carica il modello corretto e il checkpoint associato."""
    
    # Selezione del modello
    if config.model.name == "simple_cnn":
        model = SimpleCNN(num_classes=config.model.num_classes).to(device)
    
    elif config.model.name == "resnet18":
        model = get_resnet18(
            num_classes=config.model.num_classes,
            pretrained=config.model.pretrained
        ).to(device)
    
    else:
        raise ValueError(f"Modello non supportato: {config.model.name}")
    
    # Percorso dinamico del checkpoint
    checkpoint_dir = os.path.join("checkpoints", config.model.name)
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Checkpoint non trovato: {best_model_path}")

    print(f"Caricamento pesi da: {best_model_path}")
    
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    
    return model


def preprocess_image(image_path, config):
    """Preprocessa l'immagine per la predizione."""
    
    transform = transforms.Compose([
        transforms.Resize((config.data.image_size, config.data.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # batch size = 1
    
    return image


def predict(image_path):
    """Esegue la predizione su una singola immagine."""

    # Caricamento config
    with open("config/config.yaml") as f:
        config = Config(**yaml.safe_load(f))

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Si sta usando: {device}")

    # Caricamento modello
    model = load_model(config, device)

    # Preprocess
    image = preprocess_image(image_path, config).to(device)

    # Predizione
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    # Classi del Dataset
    classes = ["cat", "dog"]

    print(f"Immagine: {image_path}")
    print(f"Predicted: {classes[predicted.item()]}")

    return classes[predicted.item()]
