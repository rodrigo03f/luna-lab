"""
Benchmark EXP_001 — LOT vs AdamW fp32 na Luna Mini com dados reais.

Roda 2000 steps com cada otimizador, medindo:
    - Loss inicial e final
    - VRAM pico
    - VRAM do estado do otimizador
    - Velocidade (steps/segundo)
    - Tempo total

Uso (do container):
    cd /workspace/Luna/Luna
    python ../experiments/EXP_001_lot_vs_adamw/benchmark.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# Adiciona Luna/ ao path pra importar config/model/dataset
sys.path.insert(0, "/workspace/Luna/Luna")

from config import LunaConfig
from dataset import create_real_dataloader
from model import LunaForCausalLM
from optim.lot import LOT
from optim.lot_triton import LOTTriton

# ─── Configuração ───────────────────────────────────────────────
STEPS = 2000
BATCH_SIZE = 8
SEQ_LEN = 512
LR = 3e-4
SEED = 42
DATA_DIR = "/workspace/Luna/data/processed/megatron"
OUTPUT_FILE = "/workspace/Luna/experiments/EXP_001_lot_vs_adamw/artefatos/resultados.json"


def create_model(device: torch.device) -> LunaForCausalLM:
    """Cria modelo Mini fresh com seed fixa."""
    torch.manual_seed(SEED)
    cfg = LunaConfig.from_yaml("/workspace/Luna/Luna/configs/luna_mini.yaml")
    model = LunaForCausalLM(cfg).to(device).to(torch.bfloat16)
    return model


def create_loader() -> torch.utils.data.DataLoader:
    """Cria DataLoader com seed fixa."""
    torch.manual_seed(SEED)
    return create_real_dataloader(
        data_dir=DATA_DIR,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        num_samples_per_file=50_000,
        num_workers=0,
    )


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


def state_bytes(opt: torch.optim.Optimizer) -> int:
    """Mede bytes do estado do otimizador."""
    total = 0
    for state in opt.state.values():
        for v in state.values():
            if isinstance(v, torch.Tensor):
                total += v.numel() * v.element_size()
    return total


def run_benchmark(optimizer_name: str, device: torch.device) -> dict:
    """Roda um benchmark completo."""
    print(f"\n{'=' * 60}")
    print(f"BENCHMARK: {optimizer_name}")
    print(f"{'=' * 60}")

    # Reset completo
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(SEED)

    # Modelo + otimizador
    model = create_model(device)
    if optimizer_name == "AdamW fp32":
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    elif optimizer_name == "LOT v1.2":
        optimizer = LOTTriton(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    else:
        raise ValueError(f"Desconhecido: {optimizer_name}")

    loader = create_loader()
    train_iter = infinite(loader)

    # Aquecimento (1 step pra inicializar estado do otimizador)
    batch = next(train_iter).to(device)
    out = model(batch, labels=batch)
    out["loss"].backward()
    optimizer.step()
    optimizer.zero_grad()

    # Reset após warmup
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Mede VRAM do estado
    opt_state_mb = state_bytes(optimizer) / 1024**2

    # Loop de treino
    losses = []
    t_start = time.time()

    for step in range(1, STEPS + 1):
        batch = next(train_iter).to(device)
        out = model(batch, labels=batch)
        loss = out["loss"]
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())

        if step % 500 == 0:
            avg = sum(losses[-500:]) / 500
            print(f"  step {step:5d}/{STEPS} | loss média (últimos 500): {avg:.4f}")

    torch.cuda.synchronize()
    total_time = time.time() - t_start
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2

    # Loss final = média dos últimos 100 steps
    final_loss = sum(losses[-100:]) / 100

    result = {
        "optimizer": optimizer_name,
        "steps": STEPS,
        "final_loss": final_loss,
        "peak_vram_mb": peak_vram_mb,
        "optimizer_state_mb": opt_state_mb,
        "total_time_s": total_time,
        "steps_per_sec": STEPS / total_time,
        "losses_sample": losses[::100],  # a cada 100 steps pra plot
    }

    print(f"\n  RESULTADO:")
    print(f"    Loss final:      {final_loss:.4f}")
    print(f"    VRAM pico:       {peak_vram_mb:.1f} MB")
    print(f"    VRAM otimizador: {opt_state_mb:.2f} MB")
    print(f"    Tempo total:     {total_time:.1f}s")
    print(f"    Velocidade:      {STEPS / total_time:.1f} st/s")

    # Cleanup
    del model, optimizer, loader, train_iter
    torch.cuda.empty_cache()

    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results = []
    results.append(run_benchmark("AdamW fp32", device))
    results.append(run_benchmark("LOT v1.2", device))

    # Salva resultados
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    # Comparação final
    print(f"\n{'=' * 60}")
    print("COMPARAÇÃO FINAL")
    print(f"{'=' * 60}")

    adamw, lot = results
    print(f"{'Métrica':<20} {'AdamW':>15} {'LOT':>15} {'Melhora':>15}")
    print("-" * 65)
    print(f"{'Loss final':<20} {adamw['final_loss']:>15.4f} {lot['final_loss']:>15.4f} "
          f"{adamw['final_loss'] / lot['final_loss']:>14.3f}x")
    print(f"{'VRAM pico (MB)':<20} {adamw['peak_vram_mb']:>15.1f} {lot['peak_vram_mb']:>15.1f} "
          f"{adamw['peak_vram_mb'] / lot['peak_vram_mb']:>14.3f}x")
    print(f"{'VRAM otim (MB)':<20} {adamw['optimizer_state_mb']:>15.2f} {lot['optimizer_state_mb']:>15.2f} "
          f"{adamw['optimizer_state_mb'] / lot['optimizer_state_mb']:>14.3f}x")
    print(f"{'Velocidade (st/s)':<20} {adamw['steps_per_sec']:>15.1f} {lot['steps_per_sec']:>15.1f} "
          f"{lot['steps_per_sec'] / adamw['steps_per_sec']:>14.3f}x")

    print(f"\nResultados salvos em: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()