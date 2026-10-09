"""
Dataset para ler arquivos .bin/.idx do formato Megatron.

FORMATO .bin:
    Tokens concatenados em uint16 (2 bytes cada).
    Um arquivo pode ter múltiplos documentos, todos grudados.

FORMATO .idx:
    Header (2 int64):
        [0]: magic number (identificador do formato)
        [1]: max_seq_len (valor informativo)
    Depois do header, pares de int64:
        Para cada documento: (offset_bytes, length_bytes)

COMO O DATASET FUNCIONA:
    1. Carrega o .bin via np.memmap (não lê tudo na RAM)
    2. Na hora de amostrar, pega uma janela de seq_len tokens
       em posição aleatória do arquivo.
    3. Não respeitamos fronteiras de documento — o modelo aprende a
       lidar com tokens "grudados" de documentos diferentes.
       Isso é o padrão em treino de LLM (chamado "packing").
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset


class MegatronBinDataset(Dataset):
    """
    Dataset que lê sequências de um arquivo .bin do Megatron.

    Argumentos:
        bin_path:    caminho do arquivo .bin
        seq_len:     comprimento das sequências geradas
        num_samples: quantas sequências o dataset "finje" ter.
                     Como o arquivo tem milhões de tokens, esse número
                     só define o tamanho da epoch virtual.
    """

    def __init__(
        self,
        bin_path: str | Path,
        seq_len: int,
        num_samples: int = 100_000,
    ):
        self.bin_path = Path(bin_path)
        self.seq_len = seq_len
        self.num_samples = num_samples

        # Lê o .bin via memmap: não carrega na RAM, acessa sob demanda.
        self.tokens = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        self.total_tokens = len(self.tokens)

        if self.total_tokens < seq_len:
            raise ValueError(
                f"Arquivo {bin_path} tem apenas {self.total_tokens} tokens, "
                f"mas seq_len={seq_len} exige pelo menos {seq_len}."
            )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Sorteia uma posição inicial válida.
        # Pegamos seq_len tokens. O shift de labels é feito dentro do modelo:
        #   logits[:, :-1] vs labels[:, 1:]
        max_start = self.total_tokens - self.seq_len
        start = np.random.randint(0, max_start)

        # Lê seq_len tokens.
        chunk = self.tokens[start : start + self.seq_len]

        # Converte para int64 (o embedding do PyTorch exige long/int64).
        return torch.from_numpy(chunk.astype(np.int64))


def create_real_dataloader(
    data_dir: str | Path,
    seq_len: int,
    batch_size: int,
    num_samples_per_file: int = 100_000,
    num_workers: int = 0,
) -> DataLoader:
    """
    Cria um DataLoader lendo todos os .bin de uma pasta.

    Estratégia: concatena múltiplos arquivos .bin em um único dataset
    via ConcatDataset. Cada arquivo é um "shard" independente.
    """
    data_dir = Path(data_dir)
    bin_files = sorted(data_dir.glob("*.bin"))

    if not bin_files:
        raise FileNotFoundError(f"Nenhum arquivo .bin em {data_dir}")

    print(f"Carregando {len(bin_files)} shards de {data_dir}...")

    datasets = []
    total_tokens = 0
    for bin_path in bin_files:
        ds = MegatronBinDataset(
            bin_path=bin_path,
            seq_len=seq_len,
            num_samples=num_samples_per_file,
        )
        datasets.append(ds)
        total_tokens += ds.total_tokens

    combined = ConcatDataset(datasets)
    print(f"  Total de tokens disponíveis: {total_tokens:,}")
    print(f"  Sequências virtuais por epoch: {len(combined):,}")

    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


if __name__ == "__main__":
    # Teste rápido: carrega um shard e verifica o formato.
    print("=" * 60)
    print("TESTE DO DATASET MEGATRON")
    print("=" * 60)

    ds = MegatronBinDataset(
        "/workspace/Luna/data/processed/megatron/20336c126203.bin",
        seq_len=512,
    )
    print(f"Shard: {ds.bin_path.name}")
    print(f"Tokens disponíveis: {ds.total_tokens:,}")
    print()

    # Amostra 3 sequências e mostra os IDs
    for i in range(3):
        seq = ds[i]
        print(f"Amostra {i}: shape={seq.shape}, dtype={seq.dtype}")
        print(f"  Primeiros 20 IDs: {seq[:20].tolist()}")
        print(f"  Últimos 5 IDs:    {seq[-5:].tolist()}")
        print()

    print("✅ Dataset funcionando")