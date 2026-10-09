"""
LOT — Luna Ternary Optimizer (v1.2 — int8 blockwise).

NOTA DE NOMENCLATURA:
    O nome "Ternary" é aspiracional. A v1 usa int8 com escala por bloco,
    que ocupa O MESMO espaço (1 byte/param) mas tem 127x mais precisão.
    A v2 (kernel Triton com bit packing) vai implementar 2 bits reais.

O QUE É:
    Otimizador próprio da Luna, com estado quantizado em int8.
    Combina a regra de sinal do Lion com quantização blockwise.

POR QUE NÃO ADAMW:
    AdamW fp32: 12 bytes/param (m1 + m2 + master).
    LOT:        ~1.03 bytes/param (m_int8 + scales por bloco).
    Para Luna Grande (500M params): 6 GB -> 515 MB. 12x menos.

ALGORITMO:
    1. Desquantizar:   m_float = m_int8 / 127 * scales
    2. Update (Lion):  update = sign(β1 * m_float + (1-β1) * g)
    3. Aplicar:        p = p - lr * (update + wd * p)
    4. Novo momentum:  m_new = β2 * m_float + (1-β2) * g
    5. Requantizar:    quantize_int8_blockwise(m_new)

REFERÊNCIAS:
    - Lion: https://arxiv.org/abs/2302.06675
    - GPTQ (blockwise quant): https://arxiv.org/abs/2210.17323
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer

_EPS = 1e-8
_BLOCK_SIZE = 128
_INT8_MAX = 127.0  # range simétrico: [-127, 127]


def quantize_int8_blockwise(m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantiza um tensor float para int8 usando escalas por bloco.

    Argumentos:
        m: tensor float (momentum acumulado)

    Retorna:
        m_int8: tensor int8 com valores em [-127, 127]
        scales: tensor float com escala por bloco
    """
    flat = m.flatten().float()
    n = flat.numel()
    n_blocks = (n + _BLOCK_SIZE - 1) // _BLOCK_SIZE

    pad = n_blocks * _BLOCK_SIZE - n
    if pad > 0:
        flat_padded = torch.cat([flat, torch.zeros(pad, device=m.device)])
    else:
        flat_padded = flat

    blocks = flat_padded.view(n_blocks, _BLOCK_SIZE)

    # Escala por bloco = max(|m|)
    scales = blocks.abs().max(dim=1).values.clamp(min=_EPS)

    # Normaliza para [-1, 1] e expande para [-127, 127]
    normalized = blocks / scales.unsqueeze(1)
    quantized = torch.clamp(torch.round(normalized * _INT8_MAX), -_INT8_MAX, _INT8_MAX)
    m_int8 = quantized.flatten()[:n].to(torch.int8)

    return m_int8, scales


def dequantize_int8_blockwise(
    m_int8: torch.Tensor,
    scales: torch.Tensor,
    original_shape: torch.Size,
) -> torch.Tensor:
    """Desquantiza m_int8 de volta para float usando escalas por bloco."""
    flat = m_int8.flatten().float() / _INT8_MAX  # volta a [-1, 1]
    n = flat.numel()
    n_blocks = scales.numel()

    pad = n_blocks * _BLOCK_SIZE - n
    if pad > 0:
        flat_padded = torch.cat([flat, torch.zeros(pad, device=m_int8.device)])
    else:
        flat_padded = flat

    blocks = flat_padded.view(n_blocks, _BLOCK_SIZE)
    restored = blocks * scales.unsqueeze(1)

    return restored.flatten()[:n].view(original_shape)


class LOT(Optimizer):
    """
    Luna Ternary Optimizer (v1.2).

    Estado: momentum int8 com escalas por bloco.
    ~1.03 bytes/parâmetro, contra 12 do AdamW fp32.

    Argumentos:
        params:       parâmetros para otimizar
        lr:           learning rate (default: 3e-4)
        betas:        (beta1, beta2)
        weight_decay: L2 desacoplado (default: 0.1)
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.1,
        eps: float = _EPS,
    ):
        if lr < 0.0:
            raise ValueError(f"Learning rate inválida: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"beta1 inválido: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"beta2 inválido: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay inválido: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("LOT não suporta gradientes esparsos")

                state = self.state[p]

                # Inicialização
                if len(state) == 0:
                    n = p.numel()
                    n_blocks = (n + _BLOCK_SIZE - 1) // _BLOCK_SIZE
                    state["m_int8"] = torch.zeros(
                        n, dtype=torch.int8, device=p.device
                    )
                    state["scales"] = torch.zeros(n_blocks, device=p.device)
                    state["shape"] = p.shape

                m_int8 = state["m_int8"]
                scales = state["scales"]
                shape = state["shape"]

                # 1. Desquantiza
                m_float = dequantize_int8_blockwise(m_int8, scales, shape)

                # 2. Update (Lion)
                update_input = beta1 * m_float + (1.0 - beta1) * g.float()
                update = torch.sign(update_input)

                # 3. Aplica no peso
                p_new = p.float() - lr * (update + weight_decay * p.float())
                p.copy_(p_new.to(p.dtype))

                # 4. Novo momentum
                m_new = beta2 * m_float + (1.0 - beta2) * g.float()

                # 5. Requantiza
                m_new_int8, m_new_scales = quantize_int8_blockwise(m_new)
                state["m_int8"] = m_new_int8
                state["scales"] = m_new_scales

        return loss

    def state_bytes_per_param(self) -> float:
        """Estima bytes de estado por parâmetro (~1.03)."""
        return 1.0 + 4.0 / _BLOCK_SIZE