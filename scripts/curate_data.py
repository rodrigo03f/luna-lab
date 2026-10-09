"""
Curadoria de dados com NeMo Curator (API v1.3.0).

Uso:
    python scripts/curate_data.py

Saída:
    data/processed/curated/ (JSONL curado)
"""

import os

# ─── Silenciar log verboso ──────────────────────────────────────
os.environ.setdefault("RAY_memory_usage_threshold", "0.98")
os.environ["RAY_LOG_TO_STDERR"] = "0"
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["NEMO_CURATOR_LOG_LEVEL"] = "WARNING"
os.environ["LOGURU_LEVEL"] = "WARNING"
os.environ["COSMOS_XENNA_LOG_LEVEL"] = "WARNING"
os.environ["PYTHONWARNINGS"] = "ignore"

import glob
import logging
from pathlib import Path

for name in ["ray", "cosmos_xenna", "nemo_curator", "pandas"]:
    logging.getLogger(name).setLevel(logging.ERROR)

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.filters import ScoreFilter
from nemo_curator.stages.text.filters.heuristic import (
    NonAlphaNumericFilter,
    WordCountFilter,
)

INPUT_PATH = "/workspace/Luna/data/raw/shards"
OUTPUT_DIR = "/workspace/Luna/data/processed/curated"


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CURADORIA — Luna")
    print("=" * 60)
    print(f"  Entrada: {INPUT_PATH}")
    print(f"  Saída:   {OUTPUT_DIR}")
    print()
    print("  Rodando pipeline (pode levar alguns minutos)...")
    print()

    pipeline = Pipeline(name="luna-curation")
    pipeline.add_stage(
        JsonlReader(file_paths=INPUT_PATH, fields=["text"], blocksize="128MB")
    )
    pipeline.add_stage(
        ScoreFilter(
            filter_obj=WordCountFilter(min_words=10, max_words=100_000),
            text_field="text",
        )
    )
    pipeline.add_stage(
        ScoreFilter(
            filter_obj=NonAlphaNumericFilter(
                max_non_alpha_numeric_to_text_ratio=0.2
            ),
            text_field="text",
        )
    )
    pipeline.add_stage(JsonlWriter(path=OUTPUT_DIR))

    pipeline.run()

    outputs = sorted(glob.glob(f"{OUTPUT_DIR}/*.jsonl"))
    print()
    print("=" * 60)
    print("✅ Curadoria concluída")
    print(f"   {len(outputs)} arquivos JSONL gerados")
    print(f"   Saída: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()