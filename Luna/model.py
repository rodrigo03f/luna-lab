"""
model.py — Modelo completo Luna (Block + LM head).

Estrutura:
    LunaBlock      -> um bloco Transformer (pré-norm) com atenção + FFN ou MoE
    LunaForCausalLM -> modelo inteiro com cabeça de linguagem

DESENHO (pré-norm, como Llama/BitNet):
    x = x + Attention(RMSNorm(x))
    x = x + FFN(RMSNorm(x))

WEIGHT TYING:
    A matriz do lm_head é a MESMA do embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import Attention
from bitlinear import BitLinear
from config import LunaConfig
from experts import MoE
from ffn import SwiGLUFFN
from norm import RMSNorm


class LunaBlock(nn.Module):
    """
    Um bloco Transformer pré-norm.

    Contém:
        - Atenção causal com BitLinear e RoPE
        - FFN SwiGLU (densa) OU MoE (esparsa)
        - Duas RMSNorm
        - Duas conexões residuais
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        config: LunaConfig,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_moe = config.use_moe

        # Normalização antes da atenção.
        self.norm1 = RMSNorm(d_model)

        # Atenção causal.
        self.attention = Attention(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        # Normalização antes da FFN.
        self.norm2 = RMSNorm(d_model)

        # FFN: densa (SwiGLU) ou esparsa (MoE).
        if self.use_moe:
            self.ffn = MoE(
                d_model=d_model,
                n_experts=config.n_experts,
                top_k=config.top_k,
                d_ff_expert=config.d_ff_expert,
            )
        else:
            self.ffn = SwiGLUFFN(
                d_model=d_model,
                d_ff=config.d_ff,
                dropout=dropout,
            )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward do bloco.

        Retorna:
            out:      [B, S, D]
            aux_loss: escalar (ou None se FFN densa)
        """
        # Sub-camada 1: atenção
        x = x + self.attention(self.norm1(x))

        # Sub-camada 2: FFN/MoE
        if self.use_moe:
            ffn_out, aux_loss = self.ffn(self.norm2(x))
            x = x + ffn_out
            return x, aux_loss
        else:
            x = x + self.ffn(self.norm2(x))
            return x, None


class LunaForCausalLM(nn.Module):
    """
    Luna — modelo de linguagem causal completo.
    """

    def __init__(self, config: LunaConfig):
        super().__init__()

        self.config = config
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size
        self.use_moe = config.use_moe

        # Embedding
        self.embed = nn.Embedding(config.vocab_size, config.d_model)

        # Blocos
        self.blocks = nn.ModuleList([
            LunaBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                max_seq_len=config.max_seq_len,
                config=config,
                dropout=config.dropout,
            )
            for _ in range(config.n_layers)
        ])

        # Norm final
        self.norm = RMSNorm(config.d_model)

        # LM head (BitLinear)
        self.lm_head = BitLinear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embed.weight

        # Init
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward do modelo.

        Argumentos:
            input_ids: [batch, seq_len]
            labels:    [batch, seq_len] (opcional)

        Retorna:
            dict com "logits", "loss" (se labels) e "aux_loss" (se MoE)
        """
        batch, seq_len = input_ids.shape
        x = self.embed(input_ids)

        # Acumula aux_loss dos blocos MoE.
        total_aux_loss = 0.0
        n_moe_layers = 0

        for block in self.blocks:
            x, aux = block(x)
            if aux is not None:
                total_aux_loss = total_aux_loss + aux
                n_moe_layers += 1

        x = self.norm(x)
        logits = self.lm_head(x)

        out = {"logits": logits}

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            # Adiciona aux_loss (se MoE)
            if n_moe_layers > 0:
                avg_aux = total_aux_loss / n_moe_layers
                loss = loss + self.config.moe_aux_loss_coef * avg_aux
                out["aux_loss"] = avg_aux

            out["loss"] = loss

        return out

    def num_parameters(self, only_trainable: bool = True) -> int:
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())