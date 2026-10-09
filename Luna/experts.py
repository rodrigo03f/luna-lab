"""
experts.py — Camada Mixture of Experts (MoE) da Luna.

CONCEITO:
    MoE substitui a FFN densa por N "experts" (FFNs menores) + um
    roteador. Cada token só passa pelos top_k experts mais adequados.

POR QUE:
    Um modelo denso ativa TODOS os parâmetros por token. Um MoE ativa
    apenas top_k de n_experts. Isso permite mais capacidade total sem
    custo de compute proporcional.

    Ex: 8 experts com d_ff_expert=512 em uma d_model=512:
        Total:  8 × 3 × 512 × 512 = 6.3M params
        Ativo:  2 × 3 × 512 × 512 = 1.6M params por token
    Só 25% dos params de FFN são ativados por token.

DESIGN (aprovado no EXP_002):
    1. Roteador em nn.Linear (não BitLinear) — precisa de precisão
    2. Aux loss Switch-style (coef 0.01 no loss total)
    3. Sem capacidade máxima
    4. top_k = 2

DESAFIO DE BALANCEAMENTO:
    Sem aux loss, o roteador aprende a mandar tudo pro expert 0 (mais
    "fácil"), e os outros morrem. A aux loss penaliza distribuição
    desigual.

MATEMÁTICA DA AUX LOSS (Switch Transformer, Shazeer 2017):
    f_i = fração de tokens onde expert i está no top_k
    P_i = probabilidade média do roteador pro expert i
    aux_loss = n_experts × Σ (f_i × P_i)

    Se todos vão pro expert 0: f_0=1, P_0=1, resto 0 -> aux = 8
    Se distribui uniforme:      f_i = P_i = 1/8 -> aux = 1

REFERÊNCIAS:
    - Switch Transformer: https://arxiv.org/abs/2101.03961
    - Mixtral: https://arxiv.org/abs/2401.04088
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ffn import SwiGLUFFN


class MoE(nn.Module):
    """
    Camada Mixture of Experts.

    Argumentos:
        d_model:     dimensão do modelo
        n_experts:   número de experts
        top_k:       quantos experts ativar por token
        d_ff_expert: dimensão interna de CADA expert
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        top_k: int,
        d_ff_expert: int,
    ):
        super().__init__()
        if top_k > n_experts:
            raise ValueError(f"top_k ({top_k}) > n_experts ({n_experts})")

        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k

        # Roteador em nn.Linear (não BitLinear) — decisão de design
        # porque o roteador precisa de precisão pra distribuir bem.
        self.router = nn.Linear(d_model, n_experts, bias=False)

        # Cada expert é uma FFN SwiGLU (com BitLinear internamente).
        self.experts = nn.ModuleList([
            SwiGLUFFN(d_model, d_ff_expert)
            for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward do MoE.

        Argumentos:
            x: [batch, seq_len, d_model]

        Retorna:
            out:      [batch, seq_len, d_model]
            aux_loss: escalar (raw, sem coeficiente)
        """
        batch, seq_len, d_model = x.shape

        # Achata batch+seq pra processar todos os tokens juntos.
        x_flat = x.view(-1, d_model)  # [B*S, D]

        # ─── 1. Roteador ────────────────────────────────────────
        router_logits = self.router(x_flat)  # [B*S, N]

        # ─── 2. Top-k experts por token ─────────────────────────
        top_k_logits, top_k_indices = router_logits.topk(self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # [B*S, top_k]

        # ─── 3. Dispatch para experts ───────────────────────────
        # Loop Python (lento, mas correto). Cada expert processa só os
        # tokens roteados pra ele.
        out_flat = torch.zeros_like(x_flat)

        for expert_idx in range(self.n_experts):
            # Onde o expert_idx aparece no top_k deste token
            mask = (top_k_indices == expert_idx)  # [B*S, top_k]
            token_indices, k_indices = mask.nonzero(as_tuple=True)

            if token_indices.numel() == 0:
                continue

            # Peso do expert_idx pra cada token
            weights = top_k_weights[token_indices, k_indices]

            # Tokens deste expert
            tokens_for_expert = x_flat[token_indices]

            # Forward no expert
            expert_out = self.experts[expert_idx](tokens_for_expert)

            # Acumula (com peso)
            out_flat.index_add_(
                0, token_indices, expert_out * weights.unsqueeze(-1)
            )

        # ─── 4. Aux loss (Switch Transformer) ───────────────────
        aux_loss = self._compute_aux_loss(router_logits, top_k_indices)

        # ─── 5. Reshape ────────────────────────────────────────
        out = out_flat.view(batch, seq_len, d_model)

        return out, aux_loss

    def _compute_aux_loss(
        self,
        router_logits: torch.Tensor,
        top_k_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aux loss Switch-style (Shazeer 2017).

        f_i = fração de tokens onde i aparece no top_k
        P_i = prob média do roteador pro expert i
        aux = n_experts × Σ(f_i × P_i)
        """
        router_probs = F.softmax(router_logits.float(), dim=-1)  # [B*S, N]

        # one_hot: [B*S, top_k, N]
        one_hot = F.one_hot(top_k_indices, self.n_experts).float()
        # Soma sobre top_k e clampa em 1 (evita duplicata em empates)
        top_k_mask = one_hot.sum(dim=1).clamp(max=1.0)  # [B*S, N]

        f = top_k_mask.mean(dim=0)  # [N]
        P = router_probs.mean(dim=0)  # [N]

        aux_loss = self.n_experts * (f * P).sum()
        return aux_loss

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_experts={self.n_experts}, "
            f"top_k={self.top_k}"
        )

