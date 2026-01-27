"""
Definizione di una semplice CNN (Convolutional Neural Network) per la classificazione delle immagini.
Questo modello è pensato come baseline, ovvero un punto di partenza semplice per confronti futuri.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    CNN semplice per classificazione binaria (2 classi)
    """
    def __init__(self, num_classes=2):
        """ Costruttore del modello SimpleCNN. 
            Args: num_classes (numero di classi di output)
        """
        super(SimpleCNN, self).__init__()

        # ==================================
        # STRATO CONVOLUZIONALE 1
        # ==================================
        """ 
        Prima convoluzione:
        - Input: immagini RGB (3 canali)
        - Output: 16 canali di feature
        - Kernel size: 3x3
        - Padding: 1 (per mantenere le dimensioni spaziali)
        In questa convoluzione, il modello impara 16 filtri diversi per estrarre caratteristiche dalle immagini di input.
        Il kernel size di 3x3 è una scelta comune che permette di catturare dettagli locali nelle immagini.
        Il paddinf di 1 assicura che l'altezza e la larghezza delle feature map rimangano invariate dopo la convoluzione.
        Poi applico un'operazione di max pooling 2x2 per ridurre la dimensione spaziale delle feature map, dimezzando altezza e larghezza.
        Faccio questi per ridurre il numero di parametri nel layer successivo e per rendere il modello più robusto alle traslazioni nelle immagini.
        """
        # Conv2d: 3 input channels (RGB), 16 output channels, kernel size 3x3, padding 1
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding=1
        )
        
        # Max Pooling 2x2: dimezza altezza e larghezza
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ==================================
        # STRATO CONVOLUZIONALE 2
        # ==================================
        """
        Seconda convoluzione:
        - Input: 16 canali di feature dalla prima convoluzione
        - Output: 32 canali di feature
        - Kernel size: 3x3
        - Padding: 1
        In questa convoluzione, il modello impara 32 filtri diversi per estrarre caratteristiche più complesse dalle feature map prodotte dalla prima convoluzione.
        Anche qui, il kernel size di 3x3 e il padding di 1 sono scelte comuni per mantenere le dimensioni spaziali e catturare dettagli locali.
        """
        # Seconda convoluzione: 16 input channels, 32 output channels, kernel size 3x3, padding 1
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        # ==================================
        # STRATO FULLY CONNECTED
        # ==================================
    
        # Dopo due pooling: dimensione immagine 224 -> 112 -> 56, dimensione finale: 32*56*56
        self.fc1 = nn.Linear(
            in_features=32*56*56,
            out_features=128
        )

        # Strato finale di classificazione
        self.fc2 = nn.Linear(
            in_features=128,
            out_features=num_classes
        )

    def forward(self, x):
        # Passaggio attraverso il primo strato convoluzionale + ReLU + Pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Passaggio attraverso il secondo strato convoluzionale + ReLU + Pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Appiattimento del tensore: (B, 32, 56, 56) -> (B, 32*56*56)
        x = x.view(-1, 32 * 56 * 56)
        # Passaggio attraverso il primo strato fully connected + ReLU
        x = F.relu(self.fc1(x))
        # Passaggio attraverso il secondo strato fully connected (output)
        x = self.fc2(x)
        return x

"""
Debug rapido per verificare che il modello SimpleCNN funzioni correttamente.   
if __name__ == "__main__":
    # Test rapido del modello
    model = SimpleCNN(num_classes=2)
    print(model)

    # Creazione di un batch di immagini di esempio (batch_size=4, 3 canali, 224x224)
    sample_input = torch.randn(32, 3, 224, 224)
    sample_output = model(sample_input)
    print("Output shape:", sample_output.shape)  
"""