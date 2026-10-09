"""
Baixa uma amostra do GigaVerbo-v2 em streaming.

Objetivo: obter ~1-2 GB de texto em PT para validar o pipeline de
curadoria + tokenização ANTES de comprometer os 50B tokens finais.

O download é em streaming (não baixa o dataset inteiro). Salva em JSONL,
um documento por linha, formato que o NeMo Curator lê direto.

Uso:
    python scripts/download_sample.py

Saída:
    data/raw/gigaverbo_sample.jsonl
"""

import json
from pathlib import Path

from datasets import load_dataset

# ─── Configuração ───────────────────────────────────────────────
# ~200k docs costuma dar 1-2 GB de texto (depende do tamanho médio).
# Se o arquivo ficar muito grande ou pequeno, ajusta esse número.
MAX_DOCS = 200_000

OUTPUT_PATH = Path("/workspace/Luna/data/raw/gigaverbo_sample.jsonl")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("DOWNLOAD DE AMOSTRA — GigaVerbo-v2")
    print("=" * 60)
    print(f"\nAlvo: {MAX_DOCS:,} documentos")
    print(f"Saída: {OUTPUT_PATH}\n")

    # Streaming: não baixa tudo de uma vez, só vai iterando.
    ds = load_dataset(
        "Polygl0t/gigaverbo-v2",
        split="train",
        streaming=True,
    )

    total_chars = 0
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            if i >= MAX_DOCS:
                break

            # Salva só o campo de texto, em JSONL.
            f.write(json.dumps({"text": ex["text"]}, ensure_ascii=False) + "\n")
            total_chars += len(ex["text"])

            if (i + 1) % 10_000 == 0:
                mb = total_chars / 1024**2
                print(f"  {i + 1:,} docs baixados ({mb:.1f} MB de texto)")

    size_mb = OUTPUT_PATH.stat().st_size / 1024**2
    print(f"\n✅ Download concluído")
    print(f"   Documentos: {i + 1:,}")
    print(f"   Tamanho: {size_mb:.1f} MB")
    print(f"   Arquivo: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()