"""
Questo file definisce lo schema della configurazione.
Servce per validare il file config.yaml prima usarlo nel progetto.
"""
from pydantic import BaseModel                #BaseModel legge i dati dal file yaml e li converte in oggetti python
from typing import Literal

# ===================================
# SEZIONE DATI
# ===================================

class dataConfig(BaseModel):
    """Contiene tutti i parametri relativi al dataset."""

    root_dir: str                             # Percorso della cartella data/
    image_size: int                           # Dimnesione immagini
    batch_size: int                           # Numero di immagini per batch
    num_workers: int                          # Numero di processi per il caricamento dei dati

# ===================================
# SEZIONE MODELLO
# ===================================

class modelConfig(BaseModel):
    """Contiene tutti i parametri relativi al modello."""

    name: Literal["simple_cnn","resnet18"]              # Architettura del modello ammesso
    pretrained: bool                                            # Usare pesi pre-addestrati
    num_classes: int                                            # Numero di classi di output

# ===================================
# SEZIONE ALLENAMENTO (TRAINING)
# ===================================

class trainingconfig(BaseModel):
    """Contiene tutti i parametri relativi all'allenamento del modello."""

    learning_rate: float                                       # Tasso di apprendimento
    epochs: int                                                # Numero massimo di epoche

# ===================================
# SEZIONE EARLY STOPPING
# ===================================

class earlystoppingConfig(BaseModel):
    """Contiene tutti i parametri relativi all'early stopping."""

    enabled: bool                                             # Abilita o disabilita l'early stopping
    patience: int                                             # Numero di epoche senza miglioramento prima di fermare l'allenamento
    min_delta: float                                          # Miglioramento minimo per considerare un progresso
    target_accuracy: float                                    # Accuratezza obiettivo per fermare l'allenamento
    monitor: Literal["loss","accuracy"]                       # Metica da monitorare

# ===================================
# SEZIONE RESUME TRAINING
# ===================================
class resumeConfig(BaseModel):
    """Contiene i parametri per il resume dell'allenamento."""

    enabled: bool                                           # Abilita o disabilita il resume dell'allenamento

# ===================================
# SEZIONE ESPERIMENTI
# ===================================
class experimentConfig(BaseModel):
    """Contiene i parametri relativi agli esperimenti."""

    name: str                                                # Nome dell'esperimento

# ===================================
# CONFIGURAZIONE PRINCIPALE
# ===================================

class Config(BaseModel):
    """Contiene la configurazione principale del progetto."""

    project_name: str                                        # Nome del progetto
    seed: int                                                # Seed per la riproducibilità
    data: dataConfig                                         # Configurazione dei dati
    model: modelConfig                                       # Configurazione del modello
    training: trainingconfig                                 # Configurazione dell'allenamento
    early_stopping: earlystoppingConfig                      # Configurazione dell'early stopping
    resume: resumeConfig                                     # Configurazione del resume dell'allenamento 
    experiment: experimentConfig                            # Configurazione degli esperimenti

