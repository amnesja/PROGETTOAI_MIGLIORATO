# Dogs vs Cats Classifier 🐶🐱

Rete neurale CNN + ResNet18 per classificare immagini di cani e gatti.

## Funzionalità
- Modello SimpleCNN da zero
- Modello ResNet18 (transfer learning)
- Early stopping avanzato
- Resume training
- TensorBoard logging (loss, accuracy, confusion matrix)
- Predizione singola da immagine

## Installazione
pip install -r requirements.txt

## Esecuzione

### Training
python main.py --mode train

### Valutazione
python main.py --mode eval

### Predizione
python main.py --mode predict



