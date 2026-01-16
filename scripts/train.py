"""
Script di addestramento del modello CNN semplice o ResNet18 su un dataset di immagini.

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as Adam
import torchvision
import yaml

from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from logging import config
from config.schema import Config
from utils.dataloader import get_dataloaders
from utils.early_stopping import EarlyStopping
from models.simple_cnn import SimpleCNN
from models.resnet18 import get_resnet18
from utils.seed import set_seed

def validate(model, dataloader, criterion, device):
    """
    Esegue la validazione del modello. 
    Args:
        model: il modello da validare
        dataloader: dataloader per il set di validazione
        criterion: funzione di perdita
        device: dispositivo (CPU o GPU)
    Returns:
        val_loss: perdita media sul set di validazione
        val_accuracy: accuratezza sul set di validazione
    """
    model.eval()                            # Modalità valutazione
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():                  # Disabilita il calcolo dei gradienti
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


def plot_confusion_matrix(cm, class_names):
    """ 
        Matrice di confusione per capire quale modello è accurato e dove sbaglia con utilizzo in tensorboard.
    """

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap='Blues')

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))

    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black')

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    fig.tight_layout()
    return fig

def train(resume=False):
    # ==================================
    # CARICAMENTO CONFIGURAZIONE
    # ==================================

    with open("config/config.yaml") as f:
        config = Config(**yaml.safe_load(f))
    
    # =================================
    # CREAZIONE CARTELLA ESPERIMENTO
    # ==================================
    experiment_name = config.experiment.name
    experiment_dir = os.path.join("experiments", experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Salva la configurazione usata nell'esperimento
    with open(os.path.join(experiment_dir, "config_used.yaml"), "w") as f:
        yaml.dump(config.dict(), f)
    
    # ==================================
    # IMPOSTAZIONE SEED
    # ==================================
    
    set_seed(config.seed)
    
    # ==================================
    # TENSORBOARD WRITER
    # ==================================
    
    #writer = SummaryWriter(log_dir=f"runs/{config.model.name}")                        #vecchio modo di salvare i log
    writer = SummaryWriter(log_dir=os.path.join(experiment_dir, "tensorboard"))         # I log verranno salvati in experiments/<experiment_name>/tensorboard

    # ==================================
    # IMPOSTAZIONI DISPOSITIVO
    # ==================================
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Si sta usando: {device}")

    # ==================================
    # DATALOADER
    # ==================================
    
    train_loader, val_loader = get_dataloaders(config)

    print("Train samples:", len(train_loader.dataset))
    print("Val samples:", len(val_loader.dataset))
    print("num_workers:", config.data.num_workers)
    print(torch.cuda.is_available())
    print("Dataloader pronto")


    # ==================================
    # MODELLO
    # ==================================

    if config.model.name == "simple_cnn":
        model = SimpleCNN(
            num_classes=config.model.num_classes
        ).to(device)
    
    elif config.model.name == "resnet18":
        model = get_resnet18(
            num_classes=config.model.num_classes,
            pretrained=config.model.pretrained
        ).to(device)
    
    else:
        raise ValueError(f"Modello non supportato: {config.model.name}")

    # ================================
    # LOGGING MODELLO
    # ==================================

    dummy_input = torch.randn(
        1, 3, config.data.image_size, config.data.image_size
    ).to(device)

    writer.add_graph(model, dummy_input)


    # ==================================
    # CRITERIO E OTTIMIZZATORE
    # ==================================
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam.Adam(model.parameters(), lr=config.training.learning_rate)

    # =================================
    # RESUME TRAINING (Cartella esperimento)
    # =================================
    
    start_epoch = 0
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    if config.resume.enabled:
        """
        # Controlla se la cartella dei checkpoint esiste
        if os.path.isdir(checkpoint_dir):

            # Prendi tutti i file di checkpoint
            checkpoint_files = [f for f in os.listdir(checkpoint_dir)
                                    if f.startswith("checkpoint_epoch_") and f.endswith(".pth")
            ]
            if len(checkpoint_files) > 0:
                checkpoint_files.sort(
                    key=lambda x: int(x.split("_")[-1].split(".")[0])
                )

                latest_checkpoint = checkpoint_files[-1]
                checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
                print(f"Ripristino del training dal checkpoint: {checkpoint_path}")

                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

                start_epoch = checkpoint["epoch"]
                print(f"Ripresa dell'allenamento dalla epoch {start_epoch}")
            else:
                print("Nessun checkpoint trovato, inizio nuovo training")
        else:
            print("Nessuna cartella di checkpoint trovata, inizio nuovo training")
    else:
        print("Resume disabilitato, inizio nuovo training")
        """

        print(f"[RESUME] Ripristino del training. Controllo la cartella: {checkpoint_dir}")

        # Controllo se la cartella dei checkpoint contiene i file
        checkpoint_files = [
            f for f in os.listdir(checkpoint_dir)
            if f.startswith("checkpoint_epoch_") and f.endswith(".pth")
        ]

        if len(checkpoint_files) > 0:
            
            #Ordinamento per numero di epoch
            checkpoint_files.sort(
                key=lambda x: int(x.split("_")[-1].split(".")[0])
            )
            
            latest_checkpoint = checkpoint_files[-1]
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            print(f"[RESUME] Ripristino dal checkpoint: {checkpoint_path}")

            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"]
            print(f"[RESUME] Ripresa dell'allenamento dalla epoch {start_epoch}")
        else:
            print(f"[RESUME] Nessun checkpoint trovato in {checkpoint_dir}, inizio nuovo training")
    else:
        print("[RESUME] Resume disabilitato, inizio nuovo training")


    # ==================================
    # EARLY STOPPING
    # ==================================

    early_stopping = EarlyStopping(
        patience=config.early_stopping.patience,
        min_delta=config.early_stopping.min_delta,
    )

    # ==================================
    # CICLO DI ALLENAMENTO
    # ==================================

    for epoch in range(start_epoch,config.training.epochs):
        model.train()
        running_loss = 0.0

        # Log di immagini di esempio (solo alla prima epoca)
        if epoch == 0:
            sample_imgs, _ = next(iter(train_loader))
            img_grid = torchvision.utils.make_grid(sample_imgs[:16])
            writer.add_image("Esempi Training", img_grid)

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
        
        train_loss = running_loss / len(train_loader)

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # =================================
        # MATRICE DI CONFUSIONE
        # =================================

        all_preds = []
        all_labels = []

        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        fig = plot_confusion_matrix(cm, class_names=[str(i) for i in range(config.model.num_classes)])
        writer.add_figure("Confusion Matrix", fig, epoch+1)
        plt.close(fig)

        # =================================
        # LOGGING SU TERMINALE
        # =================================
        
        print(
            f"Epoch {epoch+1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_acc:.4f}"
        )

        # =================================
        # LOGGING TENSORBOARD
        # =================================

        writer.add_scalar("Loss/Train", train_loss, epoch+1) 
        writer.add_scalar("Loss/Validation", val_loss, epoch+1)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch+1)

        # =================================
        # SALVATAGGIO CHECKPOINT IN BASE AL MODELLO (nell esperimento)
        # =================================
        """
        checkpoint_dir = os.path.join("checkpoints", config.model.name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, checkpoint_path)
        """
        checkpoint_path = os.path.join(
            experiment_dir,"checkpoints",f"checkpoint_epoch_{epoch+1}.pth"
        )
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, checkpoint_path)

        # =================================
        # EARLY STOPPING
        # =================================
        
        early_stopping.step(val_loss,val_acc)

        if early_stopping.is_best:
            """
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Modello salvato come: {best_model_path}")
            """
            best_model_path = os.path.join(
                experiment_dir,"checkpoints","best_model.pth"
            )
            torch.save(model.state_dict(), best_model_path)
            print(f"Modello salvato come: {best_model_path}")
            
        if early_stopping.early_stop:
            print("Early stopping attivato. Interruzione dell'allenamento.")
            break

        # =================================
        # SALVATAGGIO RISULTATI ESPERIMENTO (RESULTS.JSON)
        # =================================
        results = {
            "best_val_score": early_stopping.best_score,
            "stopped_at_epoch": epoch + 1,
            "final_val_loss": val_loss,
            "final_val_accuracy": val_acc
        }
        import json

        with open(os.path.join(experiment_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=4)
        
        print(f"Risultati salvati in: {experiment_dir}/results.json")
    
    writer.close()


