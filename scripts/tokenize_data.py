"""
Tokenização + geração de arquivos .bin/.idx no formato Megatron.

NOTA: append_eod=False (o writer do NeMo 1.3.0 escreve ID 32000 em vez de
<eos> quando isso está ligado). Os limites de documento ficam no .idx.
"""

import os

# Silenciar antes de qualquer import pesado
os.environ.setdefault("RAY_memory_usage_threshold", "0.98")
os.environ["RAY_LOG_TO_STDERR"] = "0"
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore"

import glob
import logging
import sys
from pathlib import Path

for name in ["ray", "cosmos_xenna", "pandas"]:
    logging.getLogger(name).setLevel(logging.ERROR)

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer.megatron_tokenizer import (
    MegatronTokenizerWriter,
)

# ─── Silenciar loguru DEPOIS de importar (aqui é o segredo) ─────
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="ERROR")  # só erros

INPUT_DIR = "/workspace/Luna/data/processed/curated"
OUTPUT_DIR = "/workspace/Luna/data/processed/megatron"
TOKENIZER_DIR = "/workspace/Luna/Luna/tokenizer/hf_format"


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TOKENIZAÇÃO — Luna")
    print("=" * 60)
    print(f"  Entrada:   {INPUT_DIR}")
    print(f"  Saída:     {OUTPUT_DIR}")
    print(f"  Tokenizer: {TOKENIZER_DIR}")
    print()
    print("  Rodando pipeline (pode levar alguns minutos)...")
    print()

    pipeline = Pipeline(name="luna-tokenize")
    pipeline.add_stage(
        JsonlReader(file_paths=INPUT_DIR, fields=["text"], blocksize="128MB")
    )
    pipeline.add_stage(
        MegatronTokenizerWriter(
            path=OUTPUT_DIR,
            model_identifier=TOKENIZER_DIR,
            text_field="text",
            append_eod=False,
        )
    )

    pipeline.run()

    bins = sorted(glob.glob(f"{OUTPUT_DIR}/*.bin"))
    idxs = sorted(glob.glob(f"{OUTPUT_DIR}/*.idx"))

    print()
    print("=" * 60)
    print("✅ Tokenização concluída")
    print(f"   {len(bins)} arquivos .bin")
    print(f"   {len(idxs)} arquivos .idx")
    print(f"   Saída: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()