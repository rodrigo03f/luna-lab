import os
from datasets import load_dataset

# Configurações de download
DATASET_NAME = "Polygl0t/gigaverbo-v2"
SPLIT = "train"
MAX_SAMPLES = 50000  # ~2-5 GB
OUTPUT_DIR = "/workspace/Luna/data/raw"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Baixando {DATASET_NAME} (split={SPLIT}, max_samples={MAX_SAMPLES})...")

# Streaming para não baixar o dataset inteiro de uma vez
ds = load_dataset(DATASET_NAME, split=SPLIT, streaming=True)

# Salva uma amostra em JSONL
output_path = os.path.join(OUTPUT_DIR, "gigaverbo_sample.jsonl")
with open(output_path, "w", encoding="utf-8") as f:
    for i, example in enumerate(ds):
        if i >= MAX_SAMPLES:
            break
        # Salva apenas o texto (o campo pode variar — verifique o schema)
        f.write(example["text"] + "\n")
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1} documentos salvos...")

print(f"Concluído! Arquivo salvo em: {output_path}")