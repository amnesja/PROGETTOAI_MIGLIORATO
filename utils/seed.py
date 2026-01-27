# utils/seed.py

import random
import numpy as np
import torch

def set_seed(seed: int):
    """
    Imposta il seed (serve per fissare un generatore di numeri casuali) per ottenere risultati riproducibili.
    Agisce su: random, numpy, torch CPU e torch CUDA.
    Praticamente se cambio solo un parametro (es. learning rate), posso essere sicuro che la variazione nei risultati
    sia dovuta solo a quel cambiamento.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Garantisce riproducibilità con CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[SEED] Seed impostato a: {seed}")
