# 🐶🐱 Dogs vs Cats Classifier – Progetto AI

Un progetto completo per la classificazione di immagini di cani e gatti tramite reti neurali convoluzionali (CNN) e transfer learning con ResNet18.  
Include addestramento, validazione, logging avanzato, resume training, predizione e confronto dei modelli.

---

## 📌 Caratteristiche principali

### ✔️ Modelli implementati
- **SimpleCNN** – rete neurale progettata da zero  
- **ResNet18** – rete pre-addestrata tramite transfer learning

### ✔️ Funzionalità del sistema
- Dataset loader con trasformazioni e normalizzazione
- Early stopping avanzato (monitor accuracy/loss + target accuracy)
- Resume training automatico dai checkpoint
- Salvataggio automatico:
  - Checkpoint per epoca
  - `best_model.pth`
- TensorBoard logging:
  - Loss / Accuracy
  - Confusion Matrix
  - Architettura del modello
  - Immagini del dataset

### ✔️ Modalità operative
- `train` – Addestra il modello
- `eval` – Valuta il modello migliore
- `predict` – Predice classe di un'immagine singola
- `resume` – Riprende il training dall’ultimo checkpoint

---

## 📂 Struttura del progetto
```
PROGETTOAI_MIGLIORATO/
│
├── config/                 # configurazioni e schema
    ├──config.yaml
    ├──schema.py
├── models/                 # SimpleCNN + ResNet18
    ├──simple_cnn.py
    ├──resnet18.py
├── utils/                  # dataloader, early stopping, resume
    ├──dataloader.py
    ├──early_stopping
├── checkpoints/            # salvati automaticamente
├── runs/                   # per TensorBoard
├── scripts/                #script che vengono importati nel main
    ├── train.py            # training completo
    ├── evaluate.py         # valutazione sul validation/test
    ├── predict.py          # predizione su singola immagine
├── main.py                 # entry point con modalità
│
├── requirements.txt        # file librerie utilizzate
├── README.md
└── .gitignore
```

---

## ⚙️ Installazione

Assicurati di avere Python 3.9+ installato.

```bash
pip install -r requirements.txtù
```
