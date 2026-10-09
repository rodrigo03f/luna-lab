"""
RMSNorm — Root Mean Square Normalization.

O QUE FAZ:
    Normaliza as ativações de uma camada dividindo pela raiz da média
    dos quadrados (RMS). Substitui o LayerNorm em quase todos os modelos
    modernos (Llama, Mistral, BitNet, etc.).

POR QUE NÃO LAYERNORM:
    LayerNorm subtrai a média e divide pelo desvio padrão. RMSNorm só
    divide pelo RMS — não subtrai a média. Isso:
      - Remove uma operação (mais rápido)
      - Funciona igual na prática
      - Menos parâmetros (não tem bias)

MATEMÁTICA:
    Dado x (tensor de ativações):
        rms(x) = sqrt(mean(x²) + eps)
        y = x / rms(x) * gamma

    Onde gamma é um vetor aprendido (um por feature), inicializado em 1.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_EPS = 1e-6


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = _EPS):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # Parâmetro aprendido: um valor por feature, inicializado em 1.
        self.weight = nn.Parameter(torch.ones(d_model, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Guarda dtype original para restaurar no final (bf16 tem pouca
        # precisão em reduções como mean e sqrt).
        input_dtype = x.dtype

        # Cálculo interno sempre em fp32.
        x_fp32 = x.float()

        # RMS ao longo da última dimensão.
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(variance + self.eps)

        # Normaliza e escala.
        x_normalized = x_fp32 / rms * self.weight

        return x_normalized.to(input_dtype)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, eps={self.eps}"