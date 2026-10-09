"""
train.py — loop de treino da Luna.

O QUE FAZ:
    Treina a Luna a partir de uma config, com:
      - Scheduler de LR (warmup linear + cosine decay)
      - Gradient clipping (estabilidade)
      - Gradient accumulation (batch efetivo maior sem estourar VRAM)
      - Checkpoints periódicos (com resume robusto)
      - Logging de loss, LR, grad norm, VRAM, velocidade

COMO USAR:
    # Treino do zero com dados sintéticos (padrão de config)
    python train.py --config configs/luna_mini.yaml

    # Treino com dados reais (.bin/.idx do Megatron)
    # Basta ter `use_real_data: true` e `data_dir` na config.

    # Retomar de um checkpoint (pega o de maior step)
    python train.py --config configs/luna_mini.yaml --resume
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from config import LunaConfig
from dataset import create_real_dataloader
from model import LunaForCausalLM
from optim.lot_triton import LOTTriton

# ─────────────────────────────────────────────────────────────────
# SCHEDULER DE LR
# ─────────────────────────────────────────────────────────────────
def get_lr(
    step: int,
    max_steps: int,
    warmup_steps: int,
    base_lr: float,
    min_lr_ratio: float = 0.1,
) -> float:
    """
    Learning rate para o step atual.

    Regime:
        1. Warmup (step < warmup_steps): LR cresce linear de 0 até base_lr.
           Motivo: no início, os pesos são aleatórios e o gradiente é caótico.
           Subir devagar evita que o modelo "exploda" nos primeiros steps.

        2. Cosine decay (step >= warmup_steps): LR cai suavemente até
           base_lr * min_lr_ratio (10% do valor base, por padrão).
           Motivo: no final, passos menores permitem refinar a solução.
    """
    # Fase 1: warmup linear
    if step < warmup_steps:
        return base_lr * (step / max(1, warmup_steps))

    # Fase 2: cosine decay
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(progress, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * coeff)


# ─────────────────────────────────────────────────────────────────
# CHECKPOINTS
# ─────────────────────────────────────────────────────────────────
def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: LunaConfig,
    best_loss: float,
) -> None:
    """
    Salva o estado do treino em disco.

    Salva: pesos, estado do otimizador, step, config, best_loss.
    Isso permite retomar sem perder momentum do otimizador.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    from dataclasses import asdict
    config_dict = asdict(config)
    if isinstance(config_dict.get("betas"), tuple):
        config_dict["betas"] = list(config_dict["betas"])

    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "config": config_dict,
        "best_loss": best_loss,
    }

    # Salva em arquivo temporário e renomeia — evita corrupção se cair luz.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, tmp_path)
    tmp_path.replace(path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """
    Carrega um checkpoint. Restaura modelo e (se passado) otimizador.
    Retorna metadados: step, best_loss, config.
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    model.load_state_dict(ckpt["model_state"])

    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    return {
        "step": ckpt.get("step", 0),
        "best_loss": ckpt.get("best_loss", float("inf")),
        "config": ckpt.get("config", {}),
    }


# ─────────────────────────────────────────────────────────────────
# DATASET SINTÉTICO COM PADRÃO
# ─────────────────────────────────────────────────────────────────
class SyntheticDataset(torch.utils.data.Dataset):
    """
    Dataset de brinquedo com padrão APRENDÍVEL.

    Regra: cada sequência é um trecho de um ciclo de incremento +1,
    usando apenas os tokens [0, pattern_size).

    O modelo deve aprender a prever "próximo = atual + 1 (mod pattern_size)".
    Se aprender, a loss cai de log(32000) ≈ 10.4 para próximo de 0.

    Por que pattern_size=256 e não vocab_size=32000:
        Com 32000 possíveis, o modelo teria que aprender um mapeamento
        gigantesco. Com 256, o ciclo é pequeno o suficiente para que
        4 camadas aprendam com poucos steps.
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        num_samples: int = 10000,
        pattern_size: int = 256,
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.pattern_size = pattern_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = torch.randint(0, self.pattern_size, (1,)).item()
        seq = torch.arange(start, start + self.seq_len) % self.pattern_size
        return seq


def create_dataloader(
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    num_samples: int = 10000,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    """Cria um DataLoader com o dataset sintético."""
    dataset = SyntheticDataset(vocab_size, seq_len, num_samples)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def infinite_dataloader(loader):
    """Loop infinito sobre um DataLoader (reinicia ao chegar no fim)."""
    while True:
        for batch in loader:
            yield batch


# ─────────────────────────────────────────────────────────────────
# EVAL
# ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    num_batches: int = 10,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Roda N batches de eval e retorna a loss média (sem autograd)."""
    model.eval()
    losses = []
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        input_ids = batch.to(device)
        out = model(input_ids, labels=input_ids)
        losses.append(out["loss"].item())
    model.train()
    return sum(losses) / max(1, len(losses))


# ─────────────────────────────────────────────────────────────────
# OTIMIZADOR
# ─────────────────────────────────────────────────────────────────
def build_optimizer(model: nn.Module, config: LunaConfig) -> torch.optim.Optimizer:
    """
    Cria o otimizador conforme a config.

    AVISO: o LOT (Luna Ternary Optimizer) ainda NÃO está implementado.
    Enquanto isso, se a config pedir "lot", caímos em AdamW fp32 com
    aviso explícito. EXP_001 vai comparar os dois.
    """
    optimizer_name = config.optimizer

    if optimizer_name == "lot":
        print("Otimizador: LOT v2 (Luna Ternary Optimizer com kernel Triton)")
        return LOTTriton(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            weight_decay=config.weight_decay,
        )

    if optimizer_name == "adamw_fp32":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            weight_decay=config.weight_decay,
        )

    raise ValueError(f"Otimizador desconhecido: {config.optimizer}")


# ─────────────────────────────────────────────────────────────────
# LOOP DE TREINO
# ─────────────────────────────────────────────────────────────────
def train(config: LunaConfig, resume: bool = False) -> None:
    """Loop de treino completo."""

    # ─── Setup básico ────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # Seed para reprodutibilidade.
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    # ─── Modelo ──────────────────────────────────────────────────
    print("\nCriando modelo...")
    model = LunaForCausalLM(config).to(device)

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    model_dtype = dtype_map[config.dtype]
    if model_dtype != torch.float32:
        model = model.to(model_dtype)

    total_params = model.num_parameters()
    print(f"Params totais: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"Dtype de treino: {config.dtype}")

    # ─── Otimizador ──────────────────────────────────────────────
    optimizer = build_optimizer(model, config)
    print(f"Otimizador: {type(optimizer).__name__}")

    # ─── Dataloaders ─────────────────────────────────────────────
    if config.use_real_data:
        print(f"\nUsando dados REAIS de: {config.data_dir}")
        train_loader = create_real_dataloader(
            data_dir=config.data_dir,
            seq_len=config.max_seq_len,
            batch_size=config.batch_size,
            num_samples_per_file=50_000,
            num_workers=0,
        )
        eval_loader = create_real_dataloader(
            data_dir=config.data_dir,
            seq_len=config.max_seq_len,
            batch_size=config.batch_size,
            num_samples_per_file=500,
            num_workers=0,
        )
    else:
        print(f"\nUsando dados SINTÉTICOS")
        train_loader = create_dataloader(
            vocab_size=config.vocab_size,
            seq_len=config.max_seq_len,
            batch_size=config.batch_size,
            num_samples=10000,
        )
        eval_loader = create_dataloader(
            vocab_size=config.vocab_size,
            seq_len=config.max_seq_len,
            batch_size=config.batch_size,
            num_samples=500,
        )
    train_iter = infinite_dataloader(train_loader)

    # ─── Resume robusto ──────────────────────────────────────────
    start_step = 0
    best_loss = float("inf")
    ckpt_dir = Path("checkpoints")
    final_ckpt = ckpt_dir / f"{config.name}_final.pt"

    if resume:
        # Procura TODOS os checkpoints do modelo e pega o de maior step.
        # Assim o --resume retoma do último ponto salvo, não só do final.
        pattern = f"{config.name}_step_*.pt"
        candidates = sorted(
            ckpt_dir.glob(pattern),
            key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else 0,
        )
        # Inclui o final.pt se existir, como candidato
        if final_ckpt.exists():
            candidates.append(final_ckpt)

        if candidates:
            ckpt_path = candidates[-1]
            print(f"\nRetomando de {ckpt_path}...")
            meta = load_checkpoint(ckpt_path, model, optimizer)
            start_step = meta["step"]
            best_loss = meta["best_loss"]
            print(f"  step={start_step}, best_loss={best_loss:.4f}")
        else:
            print(f"\n⚠️  --resume passado mas nenhum checkpoint encontrado. Começando do zero.")

    # ─── Controle de velocidade ──────────────────────────────────
    t_start = time.time()
    t_last_log = time.time()
    steps_last_log = start_step

    # ─── Loop ────────────────────────────────────────────────────
    print(f"\nIniciando treino: step {start_step + 1} até {config.max_steps}")
    print("=" * 60)

    model.train()
    running_loss = 0.0
    running_steps = 0

    for step in range(start_step + 1, config.max_steps + 1):

        # ─── LR do step ──────────────────────────────────────────
        lr = get_lr(
            step=step,
            max_steps=config.max_steps,
            warmup_steps=config.warmup_steps,
            base_lr=config.lr,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ─── Gradient accumulation ───────────────────────────────
        optimizer.zero_grad()
        step_loss = 0.0

        for _ in range(config.gradient_accumulation):
            batch = next(train_iter).to(device)
            out = model(batch, labels=batch)
            loss = out["loss"] / config.gradient_accumulation
            loss.backward()
            step_loss += loss.item()

        # ─── Gradient clipping ───────────────────────────────────
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.grad_clip
        ).item()

        # ─── Step do otimizador ──────────────────────────────────
        optimizer.step()

        # ─── Logging ─────────────────────────────────────────────
        running_loss += step_loss
        running_steps += 1

        if step % 50 == 0 or step == start_step + 1:
            avg_loss = running_loss / running_steps

            # Velocidade ATUAL.
            now = time.time()
            dt = now - t_last_log
            ds = step - steps_last_log
            steps_per_sec = ds / dt if dt > 0 else 0.0
            t_last_log = now
            steps_last_log = step

            vram_mb = (
                torch.cuda.max_memory_allocated() / 1024**2
                if device.type == "cuda" else 0
            )

            print(
                f"step {step:5d}/{config.max_steps} | "
                f"loss {avg_loss:.4f} | "
                f"lr {lr:.2e} | "
                f"gnorm {grad_norm:.2f} | "
                f"{steps_per_sec:5.2f} st/s | "
                f"VRAM {vram_mb:.0f} MB"
            )
            running_loss = 0.0
            running_steps = 0

        # ─── Eval periódico ──────────────────────────────────────
        if step % config.eval_interval == 0:
            eval_loss = evaluate(model, eval_loader, num_batches=10, device=device)
            print(f"  ▸ eval em step {step}: loss = {eval_loss:.4f}")

            if eval_loss < best_loss:
                best_loss = eval_loss
                save_checkpoint(
                    ckpt_dir / f"{config.name}_best.pt",
                    model, optimizer, step, config, best_loss,
                )
                print(f"  ▸ novo best! salvo em {config.name}_best.pt")

        # ─── Save periódico ──────────────────────────────────────
        if step % config.save_interval == 0:
            ckpt_path = ckpt_dir / f"{config.name}_step_{step}.pt"
            save_checkpoint(ckpt_path, model, optimizer, step, config, best_loss)
            print(f"  ▸ checkpoint salvo: {ckpt_path}")

    # ─── Save final ──────────────────────────────────────────────
    save_checkpoint(final_ckpt, model, optimizer, config.max_steps, config, best_loss)
    total_time = time.time() - t_start
    print("=" * 60)
    print(f"Treino concluído em {total_time / 60:.1f} min")
    print(f"Checkpoint final: {final_ckpt}")
    print(f"Melhor loss de eval: {best_loss:.4f}")


# ─────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Treino da Luna")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Caminho para o arquivo YAML de config",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Retoma do checkpoint de maior step disponível",
    )
    args = parser.parse_args()

    config = LunaConfig.from_yaml(args.config)
    print(config.summary())
    print()

    train(config, resume=args.resume)


if __name__ == "__main__":
    main()