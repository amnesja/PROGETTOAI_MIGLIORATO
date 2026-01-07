"""
Implementazione dell'Early Stopping.
Ferma l'addestramento se la performance sul set di validazione non migliora
dopo un certo numero di epoche (patience) o se viene raggiunta una certa accuratezza obiettivo.
"""

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, target_accuracy=None, monitor="accuracy"):
        """
        Early stopping avanzato.

        Args:
            patience (int): epoche senza miglioramento prima di fermare.
            min_delta (float): miglioramento minimo.
            target_accuracy (float): se raggiunta, ferma l'allenamento.
            monitor (str): 'accuracy' o 'loss'.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.target_accuracy = target_accuracy
        self.monitor = monitor

        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.is_best = False

    def step(self, val_loss, val_acc):
        """
        Aggiorna lo stato dell'early stopping.

        Args:
            val_loss (float): validazione loss
            val_acc (float): validazione accuracy
        """

        self.is_best = False

        # Se monitoriamo accuracy
        if self.monitor == "accuracy":
            metric = val_acc
            improvement = metric - (self.best_score if self.best_score is not None else -1)

            # Target raggiunto
            if self.target_accuracy is not None and val_acc >= self.target_accuracy:
                self.early_stop = True
                self.is_best = True
                return

        # Se monitoriamo loss
        elif self.monitor == "loss":
            metric = -val_loss  # invertiamo perché loss minore è migliore
            improvement = metric - (self.best_score if self.best_score is not None else float("-inf"))

        else:
            raise ValueError("Monitor deve essere 'accuracy' o 'loss'.")

        # Prima iterazione
        if self.best_score is None:
            self.best_score = metric
            self.is_best = True
            return

        # Miglioramento significativo
        if improvement > self.min_delta:
            self.best_score = metric
            self.counter = 0
            self.is_best = True

        else:  # nessun miglioramento
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True