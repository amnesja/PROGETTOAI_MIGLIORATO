"""
Script di test per verificare che la configurazione YAML
rispetti lo schema definito in schema.py
"""

import yaml
from config.schema import Config

# Apriamo il file di configurazione CORRETTO
with open("config/config.yaml", "r") as f:
    raw_config = yaml.safe_load(f)

# Controllo di sicurezza (didattico)
if raw_config is None:
    raise ValueError("Il file config.yaml è vuoto o non valido")

# Validazione tramite Pydantic
config = Config(**raw_config)

print("Configurazione valida!")
print(config)