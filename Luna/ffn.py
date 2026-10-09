"""
FFN — Feed-Forward Network com SwiGLU.

O QUE FAZ:
    Aplicação ponto-a-ponto (não mistura informação entre tokens — isso é
    o papel da atenção). Cada token passa por uma MLP independente.

POR QUE SWIGLU E NÃO RELU:
    SwiGLU usa um "portão" (gate) que multiplica outro ramo (up), permitindo
    que a rede aprenda a selecionar quanto de cada feature passar adiante.

    Comparado com FFN ReLU clássica (2 matrizes), SwiGLU tem 3 matrizes,
    mas é mais expressivo com o mesmo número total de parâmetros.

MATEMÁTICA:
    Dado x [B, S, D]:
        gate = BitLinear(D → d_ff)(x)
        up   = BitLinear(D → d_ff)(x)
        hidden = silu(gate) * up
        out  = BitLinear(d_ff → D)(hidden)

    Onde silu(x) = x * sigmoid(x)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitlinear import BitLinear


class SwiGLUFFN(nn.Module):
    """
    Feed-Forward Network com ativação SwiGLU.

    Argumentos:
        d_model: dimensão do modelo (entrada e saída)
        d_ff:    dimensão interna da FFN
        dropout: taxa de dropout (0 = desligado)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        # Duas projeções "de entrada": uma vira o gate, outra o conteúdo.
        # Ambas vão de d_model para d_ff.
        self.gate_proj = BitLinear(d_model, d_ff, bias=False)
        self.up_proj = BitLinear(d_model, d_ff, bias=False)

        # Projeção "de saída": volta de d_ff para d_model.
        self.down_proj = BitLinear(d_ff, d_model, bias=False)

        # Dropout na saída (0 = sem dropout).
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward da FFN.

        Argumentos:
            x: [batch, seq_len, d_model]

        Retorna:
            out: [batch, seq_len, d_model]
        """
        # 1. Projeta em duas direções.
        gate = self.gate_proj(x)   # [B, S, d_ff]
        up = self.up_proj(x)       # [B, S, d_ff]

        # 2. SwiGLU: silu(gate) * up
        #    silu(x) = x * sigmoid(x)
        hidden = F.silu(gate) * up  # [B, S, d_ff]

        # 3. Projeta de volta.
        out = self.down_proj(hidden)  # [B, S, d_model]

        # 4. Dropout.
        out = self.dropout(out)

        return out

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_ff={self.d_ff}"