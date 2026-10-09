"""
LOT Triton — versão com kernel fundido (v2).

O QUE MUDA vs LOT v1.2 (PyTorch):
    A v1.2 faz ~20 operações PyTorch por tensor de parâmetro:
        flatten -> pad -> reshape -> abs -> max -> clamp -> div ->
        round -> clamp -> to(int8) -> to(float) -> mul -> unsqueeze ...
    Cada operação é um kernel CUDA separado. Overhead massivo.

    A v2 faz TUDO em um único kernel Triton por bloco. Um launch,
    todos os cálculos dentro. Deve recuperar 90-95% da velocidade.

ESTRATÉGIA:
    Grid = n_elements / BLOCK_SIZE (128 elementos por programa).
    Cada programa Triton processa um bloco de 128 elementos:
        1. Carrega m_int8, scale, g, p
        2. Desquantiza m
        3. Update (Lion) e aplica em p
        4. Calcula m_new
        5. Requantiza
        6. Escreve p, m_int8, scale
    Tudo em uma passada. Sem intermediários em VRAM.

REQUISITOS:
    - Triton >= 3.0
    - CUDA
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.optim import Optimizer

_INT8_MAX = 127.0
_BLOCK_SIZE = 128
_EPS = 1e-8


@triton.jit
def _lot_kernel(
    p_ptr,             # pesos (bf16/fp16/fp32)
    g_ptr,             # gradientes (mesmo dtype de p)
    m_ptr,             # momentum int8
    scale_ptr,         # escalas (float32) por bloco
    n_elements,        # total de elementos no tensor
    lr,                # learning rate
    beta1,             # beta1 do Lion
    beta2,             # beta2 do Lion
    weight_decay,      # weight decay desacoplado
    BLOCK_SIZE: tl.constexpr,
    INT8_MAX: tl.constexpr,
    EPS: tl.constexpr,
):
    """
    Kernel do update LOT.

    Grid = ceil(n_elements / BLOCK_SIZE).
    Cada programa processa um bloco inteiro de BLOCK_SIZE elementos.

    O BLOCK_SIZE deve ser igual ao tamanho do bloco de quantização (128),
    então cada programa tem exatamente 1 escala associada.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # ─── 1. Carrega tudo de uma vez ─────────────────────────────
    p = tl.load(p_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    m_int8 = tl.load(m_ptr + offsets, mask=mask, other=0).to(tl.float32)
    scale = tl.load(scale_ptr + pid).to(tl.float32)

    # ─── 2. Desquantiza momentum ────────────────────────────────
    # m_float = m_int8 / INT8_MAX * scale
    m_float = m_int8 / INT8_MAX * scale

    # ─── 3. Update (Lion): u = sign(β1 * m + (1-β1) * g) ────────
    update_input = beta1 * m_float + (1.0 - beta1) * g
    # Triton não tem `sign` pra float direto — fazemos com where.
    update = tl.where(update_input > 0, 1.0, tl.where(update_input < 0, -1.0, 0.0))

    # ─── 4. Aplica no peso ──────────────────────────────────────
    # p_new = p - lr * (update + weight_decay * p)
    p_new = p - lr * (update + weight_decay * p)
    tl.store(p_ptr + offsets, p_new.to(p_ptr.dtype.element_ty), mask=mask)

    # ─── 5. Novo momentum ───────────────────────────────────────
    m_new = beta2 * m_float + (1.0 - beta2) * g

    # ─── 6. Requantiza ──────────────────────────────────────────
    # No bloco de 128, a escala é o max(|m_new|) dentro deste programa.
    # Precisamos do absmax entre os 128 elementos do bloco.
    m_new_absmax = tl.max(tl.abs(m_new), axis=0)
    m_new_scale = tl.maximum(m_new_absmax, EPS)

    # Normaliza e quantiza para [-127, 127]
    m_new_normalized = m_new / m_new_scale
    m_new_quantized = m_new_normalized * INT8_MAX
    # Clamp e arredonda
    m_new_clamped = tl.minimum(tl.maximum(m_new_quantized, -INT8_MAX), INT8_MAX)
    # Arredondamento: para trás se fracionário (int() trunca)
    # Adicionamos 0.5*sign para arredondar "round half away from zero"
    m_new_rounded = tl.where(
        m_new_clamped >= 0,
        (m_new_clamped + 0.5).to(tl.int32),
        (m_new_clamped - 0.5).to(tl.int32),
    )
    m_new_int8 = m_new_rounded.to(tl.int8)

    # ─── 7. Escreve de volta ────────────────────────────────────
    tl.store(m_ptr + offsets, m_new_int8, mask=mask)
    tl.store(scale_ptr + pid, m_new_scale)


### BLOCO 2
class LOTTriton(Optimizer):
    """
    LOT v2 — kernel Triton fundido.

    Interface idêntica ao LOT v1.2 (mesmos hiperparâmetros, mesmo
    comportamento matemático). A diferença é que o update inteiro roda
    em um único kernel Triton por bloco, em vez de dezenas de operações
    PyTorch.

    Uso:
        optimizer = LOTTriton(model.parameters(), lr=3e-4)
        ...
        loss.backward()
        optimizer.step()
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
        """
        Executa um passo de update. Usa o kernel Triton quando possível;
        cai pra PyTorch puro em CPU ou tensores não-contíguos.
        """
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

                # ─── Inicialização ──────────────────────────────────
                if len(state) == 0:
                    n = p.numel()
                    n_blocks = (n + _BLOCK_SIZE - 1) // _BLOCK_SIZE
                    state["m_int8"] = torch.zeros(
                        n, dtype=torch.int8, device=p.device
                    )
                    state["scales"] = torch.zeros(n_blocks, device=p.device)

                m_int8 = state["m_int8"]
                scales = state["scales"]
                n_elements = p.numel()

                # ─── Caminho Triton (CUDA + contiguous) ─────────────
                if (
                    p.is_cuda
                    and p.is_contiguous()
                    and g.is_contiguous()
                    and m_int8.is_contiguous()
                    and scales.is_contiguous()
                ):
                    grid = (triton.cdiv(n_elements, _BLOCK_SIZE),)
                    _lot_kernel[grid](
                        p,
                        g,
                        m_int8,
                        scales,
                        n_elements,
                        lr,
                        beta1,
                        beta2,
                        weight_decay,
                        BLOCK_SIZE=_BLOCK_SIZE,
                        INT8_MAX=_INT8_MAX,
                        EPS=_EPS,
                    )
                else:
                    # ─── Fallback PyTorch (CPU ou não-contíguo) ─────
                    self._step_pytorch_fallback(
                        p, g, m_int8, scales, lr, beta1, beta2, weight_decay
                    )

        return loss

    @torch.no_grad()
    def _step_pytorch_fallback(
        self,
        p: torch.Tensor,
        g: torch.Tensor,
        m_int8: torch.Tensor,
        scales: torch.Tensor,
        lr: float,
        beta1: float,
        beta2: float,
        weight_decay: float,
    ) -> None:
        """
        Fallback em PyTorch puro. Mesma matemática do kernel Triton, mas
        sem os benefícios de performance. Usado só em CPU ou tensores
        não-contíguos.
        """
        # Importa as funções da v1.2 para não duplicar a lógica.
        from optim.lot import (
            dequantize_int8_blockwise,
            quantize_int8_blockwise,
        )

        shape = p.shape
        m_float = dequantize_int8_blockwise(m_int8, scales, shape)
        update_input = beta1 * m_float + (1.0 - beta1) * g.float()
        update = torch.sign(update_input)
        p_new = p.float() - lr * (update + weight_decay * p.float())
        p.copy_(p_new.to(p.dtype))
        m_new = beta2 * m_float + (1.0 - beta2) * g.float()
        m_new_int8, m_new_scales = quantize_int8_blockwise(m_new)
        m_int8.copy_(m_new_int8)
        scales.copy_(m_new_scales)