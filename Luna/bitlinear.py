"""
BitLinear — quantização ternária de pesos (inspirado em BitNet b1.58).

O QUE FAZ:
    Em vez de guardar pesos em bf16 (16 bits), guarda em {-1, 0, +1} (2 bits).
    Isso reduz a VRAM dos pesos em ~8x, e prepara o terreno para o LOT.

COMO FUNCIONA (matemática):
    Dado um peso W (float), calculamos:
        gamma = mean(|W|)                 # escala = magnitude média
        W_q = clamp(round(W / gamma), -1, 1)  # ternário
    Então reconstruímos:
        W_ternário = W_q * gamma          # volta à escala original, mas quantizado

    No forward, usamos W_ternário no lugar de W.

O PROBLEMA DO BACKWARD:
    `round()` tem derivada zero em quase todo lugar. Se usássemos a derivada
    real, o gradiente não fluiria para o peso e ele nunca atualizaria.
    Solução: STRAIGHT-THROUGH ESTIMATOR (STE).
        No forward:  usa W_ternário (quantizado)
        No backward: finge que a quantização foi identidade (passa o gradiente direto)

    É uma aproximação, mas funciona na prática — e é o que todos os papers
    de quantização fazem.

REFERÊNCIAS (para estudo, não para cópia):
    - BitNet b1.58 (Wang et al., 2023): https://arxiv.org/abs/2402.17764
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Constante para evitar divisão por zero quando todos os pesos forem 0
# (acontece em inicializações estranhas ou camadas mortas).
_EPS_SCALE = 1e-8


def quantize_weights_ternary(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantiza um tensor de pesos para valores ternários {-1, 0, +1},
    retornando também o fator de escala usado.

    Argumentos:
        weight: tensor de pesos (qualquer shape)

    Retorna:
        w_scaled: pesos ternários já multiplicados pela escala (mesmo shape)
        gamma:    a escala usada (escalar por tensor)
    """
    # Escala = magnitude média absoluta dos pesos.
    # Isso garante que a maioria dos pesos caia em {-1, 0, 1} com boa distribuição.
    gamma = weight.abs().mean().clamp(min=_EPS_SCALE)

    # Normaliza e arredonda: valores viram aproximadamente -1, 0, ou 1.
    w_q = torch.clamp(torch.round(weight / gamma), -1.0, 1.0)

    # Re-escala: agora w_scaled tem valores em {-gamma, 0, +gamma}.
    # Isso é o que entra no forward.
    w_scaled = w_q * gamma

    return w_scaled, gamma

# ─────────────────────────────────────────────────────────────────
# STE (Straight-Through Estimator)
# Um autograd.Function que permite "fingir" que a quantização é
# diferenciável no backward, mesmo não sendo.
# ─────────────────────────────────────────────────────────────────
class _TernarySTE(torch.autograd.Function):
    """
    Forward: quantiza para ternário {-1, 0, +1}.
    Backward: passa o gradiente direto (identidade).

    Isso engana o autograd: ele "acha" que a operação de quantização
    é linear, então o gradiente flui normalmente para o peso original.
    """

    @staticmethod
    def forward(ctx, weight: torch.Tensor) -> torch.Tensor:
        # No forward, fazemos a quantização de verdade.
        w_scaled, _ = quantize_weights_ternary(weight)
        return w_scaled

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # No backward, simplesmente devolvemos o gradiente como veio.
        # É uma mentira útil — funciona na prática.
        return grad_output


class BitLinear(nn.Module):
    """
    Camada linear com pesos ternários.

    É a substituição direta de nn.Linear na arquitetura da Luna.
    Na maior parte do tempo, o comportamento é idêntico ao Linear,
    exceto que os pesos são quantizados antes do forward.

    Argumentos:
        in_features:  tamanho da entrada
        out_features: tamanho da saída
        bias:         se True, adiciona bias (default: False — bias em bf16 é barato,
                      mas BitNet geralmente não usa bias)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Pesos ficam em fp32 no otimizador, mas o forward usa a versão ternária.
        # Isso é essencial: o mestre em fp32 é que recebe gradiente e é atualizado.
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.float32)
        )

        # Inicialização estilo Transformer: Kaiming com fan_in.
        # Isso mantém a variância das ativações estável ao longo das camadas.
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

        # Bias opcional (normalmente desligado na Luna)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward:
            1. Quantiza os pesos para {-1, 0, +1} (com STE no backward)
            2. Faz a multiplicação linear normal com os pesos quantizados
        """
        # Aplica o STE: no forward, o resultado é o peso quantizado;
        # no backward, o gradiente volta como se fosse o peso original.
        w_ternary = _TernarySTE.apply(self.weight)

        # Linear: y = x @ W^T + b
        # Usa F.linear que é otimizado (chama cublas ou similar).
        return F.linear(x, w_ternary, self.bias)

    def extra_repr(self) -> str:
        # Aparece bonitinho quando você imprime o modelo.
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"ternary=True"
        )

    