"""
RoPE — Rotary Position Embeddings (versão sem complexos).

MUDANÇA v2:
    A versão anterior guardava frequências como complex64. Isso quebrava
    silenciosamente quando o modelo era castado para bf16 — o PyTorch
    descartava a parte imaginária e o RoPE virava "multiplicar por cos".

    Esta versão guarda cos e sin separados, em fp32, e faz a rotação
    com matemática explícita. Resultado:
      - Compatível com qualquer dtype (bf16, fp16, fp32)
      - Mais rápido (sem view_as_complex/view_as_real)
      - Sem surpresas quando o modelo é castado

MATEMÁTICA (inalterada):
    Para cada par (x_2i, x_2i+1) na posição m:
        x'_2i    = x_2i * cos(m·θ_i) - x_2i+1 * sin(m·θ_i)
        x'_2i+1  = x_2i * sin(m·θ_i) + x_2i+1 * cos(m·θ_i)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def precompute_freqs(
    head_dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pré-calcula cos e sin do RoPE para todas as posições.

    Retorna:
        cos: [max_seq_len, head_dim // 2] em fp32
        sin: [max_seq_len, head_dim // 2] em fp32
    """
    # θ_i = base^(-2i/head_dim) para i = 0, 2, 4, ..., head_dim-2
    i = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    theta = 1.0 / (base ** (i / head_dim))

    # Posições m
    m = torch.arange(max_seq_len, dtype=torch.float32, device=device)

    # freqs[m, i] = m * θ_i
    freqs = torch.outer(m, theta)

    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Aplica RoPE em um tensor (Q ou K).

    Argumentos:
        x:   [batch, n_heads, seq_len, head_dim]
        cos: [seq_len, head_dim // 2] — DEVE estar em fp32
        sin: [seq_len, head_dim // 2] — DEVE estar em fp32

    Retorna:
        x_rot: mesmo shape de x, mesmo dtype.
    """
    input_dtype = x.dtype

    # Trabalha em fp32 internamente (cos/sin já estão em fp32).
    x = x.float()

    # Separa pares: dimensões pares (x1) e ímpares (x2).
    # x1, x2: [batch, n_heads, seq_len, head_dim // 2]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    # cos e sin precisam do shape [1, 1, seq_len, head_dim // 2] pra broadcast.
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Rotação 2D em cada par.
    x1_rot = x1 * cos - x2 * sin
    x2_rot = x1 * sin + x2 * cos

    # Interleave de volta: [.., head_dim//2] + [.., head_dim//2] → [.., head_dim]
    # torch.stack na última dim, depois flatten.
    out = torch.stack([x1_rot, x2_rot], dim=-1).flatten(-2)

    return out.to(input_dtype)


class RoPE(nn.Module):
    """
    Aplica RoPE em Q e K.

    Argumentos:
        head_dim:    dimensão por cabeça
        max_seq_len: comprimento máximo de sequência
        base:        base das frequências (10000 padrão)
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        base: float = 10000.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        cos, sin = precompute_freqs(head_dim, max_seq_len, base)
        # Registra como buffers (não treináveis, mas acompanham o modelo).
        # persistent=False → não são salvos no state_dict.
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        q, k: [batch, n_heads, seq_len, head_dim]
        """
        seq_len = q.shape[2]

        # Recorta cos/sin pro tamanho da sequência.
        # `float()` garante fp32 mesmo se o modelo foi castado (mas cos/sin
        # já são fp32 por construção — isso é redundância de segurança).
        cos = self.cos_cached[:seq_len].float()
        sin = self.sin_cached[:seq_len].float()

        q_rot = apply_rope(q, cos, sin)
        k_rot = apply_rope(k, cos, sin)

        return q_rot, k_rot

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, max_seq_len={self.max_seq_len}, base={self.base}"