"""
Treina o tokenizer BPE 32k da Luna numa mistura de PT + código + inglês.

Mistura:
    - 50% português (GigaVerbo-v2)
    - 30% código (The Stack smol)
    - 20% inglês técnico (Wikipedia EN)

Algoritmo: Byte-Level BPE (zero OOV, padrão em LLMs decoder-only).
Vocab: 32.000 tokens.
Special tokens: <pad>, <bos>, <eos>, <unk>.

Uso:
    python scripts/train_tokenizer.py

Saída:
    Luna/tokenizer/luna_tokenizer.json
"""

import os
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

# ─── Configurações ──────────────────────────────────────────────
VOCAB_SIZE = 32000
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

# Quantos documentos usar de cada fonte pra treinar o tokenizer.
# Não precisa ser muito — 100k docs por fonte já dá uma boa amostra.
N_PT = 100_000
N_CODE = 60_000
N_EN = 40_000

OUTPUT_PATH = Path("/workspace/Luna/Luna/tokenizer/luna_tokenizer.json")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def iter_pt():
    """Itera sobre documentos em português (GigaVerbo-v2)."""
    print("  Carregando GigaVerbo (streaming)...")
    ds = load_dataset("Polygl0t/gigaverbo-v2", split="train", streaming=True)
    for i, ex in enumerate(ds):
        if i >= N_PT:
            break
        yield ex["text"]


def iter_code():
    """Itera sobre código (CodeParrot Clean — público, ungated)."""
    print("  Carregando CodeParrot Clean (streaming)...")
    ds = load_dataset(
        "codeparrot/codeparrot-clean", split="train", streaming=True
    )
    for i, ex in enumerate(ds):
        if i >= N_CODE:
            break
        # O campo de código é 'content'.
        yield ex["content"]


def iter_en():
    """Itera sobre inglês técnico (Wikipedia EN)."""
    print("  Carregando Wikipedia EN (streaming)...")
    ds = load_dataset(
        "wikimedia/wikipedia", "20231101.en", split="train", streaming=True
    )
    for i, ex in enumerate(ds):
        if i >= N_EN:
            break
        yield ex["text"]


def batch_iterator(batch_size=1000):
    """
    Junta as três fontes e yielda em lotes.

    A lib tokenizers treina mais rápido com lotes do que documento a documento.
    """
    buffer = []

    # PT
    for text in iter_pt():
        buffer.append(text)
        if len(buffer) >= batch_size:
            yield buffer
            buffer = []

    # Código
    for text in iter_code():
        buffer.append(text)
        if len(buffer) >= batch_size:
            yield buffer
            buffer = []

    # Inglês
    for text in iter_en():
        buffer.append(text)
        if len(buffer) >= batch_size:
            yield buffer
            buffer = []

    # Último lote (se sobrar)
    if buffer:
        yield buffer


def main():
    print("=" * 60)
    print("TREINO DO TOKENIZER BPE 32k — Luna")
    print("=" * 60)

    # ─── 1. Cria tokenizer Byte-Level BPE ────────────────────────
    print("\n[1/4] Criando tokenizer Byte-Level BPE...")
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # ─── 2. Configura trainer ────────────────────────────────────
    print("[2/4] Configurando trainer...")
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )

    # ─── 3. Treina ───────────────────────────────────────────────
    print("[3/4] Treinando tokenizer (pode levar 10-20 min)...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    # ─── 4. Salva ────────────────────────────────────────────────
    print(f"\n[4/4] Salvando em {OUTPUT_PATH}...")
    tokenizer.save(str(OUTPUT_PATH))

    # ─── Teste rápido ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TESTE RÁPIDO DO TOKENIZER")
    print("=" * 60)

    test_pt = "A Luna é uma inteligência artificial pessoal que ajuda a programar."
    test_code = "def luna_forward(x):\n    return x @ self.weight.T + self.bias"

    for label, text in [("PT", test_pt), ("Código", test_code)]:
        enc = tokenizer.encode(text)
        print(f"\n[{label}]")
        print(f"  Texto: {text[:60]}")
        print(f"  Tokens: {len(enc.ids)}")
        print(f"  Primeiros 10: {enc.tokens[:10]}")

    print(f"\n✅ Tokenizer salvo em: {OUTPUT_PATH}")
    print(f"   Vocab size efetivo: {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()