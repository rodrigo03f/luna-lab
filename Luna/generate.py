"""
Geração de texto com a Luna.

Carrega um checkpoint e gera texto a partir de prompts.
Usa temperature + top_k sampling (greedy é muito repetitivo em modelos
pequenos).

Uso:
    python generate.py --config configs/luna_mini.yaml \
        --checkpoint checkpoints/luna_mini_final.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from config import LunaConfig
from model import LunaForCausalLM


@torch.no_grad()
def generate(
    model: LunaForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int = 60,
    temperature: float = 0.8,
    top_k: int = 50,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """
    Geração autorregressiva com temperature + top_k.

    Argumentos:
        input_ids: [1, seq_len] — prompt já tokenizado
        max_new_tokens: quantos tokens novos gerar
        temperature: > 1 = mais criativo, < 1 = mais conservador
        top_k: só amostra dos K tokens mais prováveis
    """
    model.eval()
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        # Trunca ao max_seq_len se necessário
        context = generated[:, -model.config.max_seq_len :]

        out = model(context)
        # Logits do último token
        logits = out["logits"][:, -1, :] / temperature

        # Top-k: zera tudo abaixo do k-ésimo maior
        if top_k > 0:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            threshold = values[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < threshold,
                torch.full_like(logits, float("-inf")),
                logits,
            )

        # Softmax + amostragem
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=1)

    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}\n")

    # Carrega config e modelo
    config = LunaConfig.from_yaml(args.config)
    model = LunaForCausalLM(config).to(device).to(torch.bfloat16)

    # Carrega checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {ckpt_path}")

    print(f"Carregando checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    print(f"  step={ckpt.get('step', '?')}, best_loss={ckpt.get('best_loss', '?'):.4f}\n")

    # Carrega tokenizer
    tokenizer = Tokenizer.from_file("tokenizer/luna_tokenizer.json")

    # Prompts de teste
    prompts = [
        "A Luna é uma inteligência artificial",
        "O Brasil é um país",
        "def somar(a, b):",
        "Era uma vez",
        "",
    ]

    print("=" * 70)
    print(f"GERAÇÃO — temperature={args.temperature}, top_k={args.top_k}")
    print("=" * 70)

    for prompt in prompts:
        # Tokeniza (com BOS no início)
        ids = [1] + tokenizer.encode(prompt).ids  # 1 = <bos>
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)

        # Gera
        output_ids = generate(
            model,
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
        )

        # Decodifica só os tokens novos
        new_ids = output_ids[0, len(ids):].tolist()
        new_text = tokenizer.decode(new_ids)

        print(f"\n📝 Prompt: {prompt!r}")
        print(f"   → {new_text!r}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()