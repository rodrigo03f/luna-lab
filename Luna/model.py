"""
Luna — modelo completo (block + cabeça de linguagem).

Este arquivo junta todas as peças testadas:
    - BitLinear    (pesos ternários)
    - RMSNorm      (normalização)
    - RoPE         (posição relativa)
    - Attention    (atenção causal)
    - SwiGLUFFN    (feed-forward com portão)

Estrutura:
    LunaBlock      → um bloco Transformer (pré-norm)
    LunaForCausalLM → modelo inteiro com cabeça de linguagem

DESENHO (pré-norm, como Llama/BitNet):
    x = x + Attention(RMSNorm(x))
    x = x + FFN(RMSNorm(x))

POR QUE PRÉ-NORM EM VEZ DE PÓS-NORM:
    Pós-norm (original) normaliza DEPOIS da subcamada. Isso é instável em
    modelos profundos. Pré-norm normaliza ANTES, e o residual flui limpo.
    É o padrão desde GPT-2 e todos os modelos modernos usam.

WEIGHT TYING:
    A matriz do lm_head é a MESMA do embedding. Isso economiza
    vocab_size × d_model parâmetros (no nosso caso, 8.2M — quase
    o modelo inteiro). Llama, GPT-2, Mistral, todos usam.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from attention import Attention
from bitlinear import BitLinear
from config import LunaConfig
from ffn import SwiGLUFFN
from norm import RMSNorm


class LunaBlock(nn.Module):
    """
    Um bloco Transformer pré-norm.

    Contém:
        - Atenção causal com BitLinear e RoPE
        - FFN SwiGLU com BitLinear
        - Duas RMSNorm (uma antes da atenção, uma antes da FFN)
        - Duas conexões residuais

    Argumentos:
        d_model:     dimensão do modelo
        n_heads:     número de cabeças de atenção
        d_ff:        dimensão interna da FFN
        max_seq_len: comprimento máximo de sequência
        dropout:     taxa de dropout
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Normalização antes da atenção (pré-norm).
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

        # Feed-forward SwiGLU.
        self.ffn = SwiGLUFFN(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward do bloco.

        Argumentos:
            x: [batch, seq_len, d_model]

        Retorna:
            out: [batch, seq_len, d_model]
        """
        # ─── Sub-camada 1: atenção ─────────────────────────────
        # x = x + Attention(RMSNorm(x))
        x = x + self.attention(self.norm1(x))

        # ─── Sub-camada 2: FFN ─────────────────────────────────
        # x = x + FFN(RMSNorm(x))
        x = x + self.ffn(self.norm2(x))

        return x


class LunaForCausalLM(nn.Module):
    """
    Luna — modelo de linguagem causal completo.

    Argumentos:
        config: objeto LunaConfig com todos os hiperparâmetros

    Atributos:
        embed:   tabela de embeddings [vocab_size, d_model]
        blocks:  lista de LunaBlock
        norm:    RMSNorm final (antes do lm_head)
        lm_head: BitLinear que projeta d_model → vocab_size
                 (com weight tying, compartilha o peso com o embed)
    """

    def __init__(self, config: LunaConfig):
        super().__init__()

        self.config = config
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size

        # ─── Embedding ─────────────────────────────────────────
        # Tabela que converte índices de token em vetores densos.
        self.embed = nn.Embedding(config.vocab_size, config.d_model)

        # ─── Blocos Transformer ────────────────────────────────
        # Empilha N blocos idênticos.
        self.blocks = nn.ModuleList([
            LunaBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff if not config.use_moe else config.d_ff_expert,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout,
            )
            for _ in range(config.n_layers)
        ])

        # ─── Normalização final ────────────────────────────────
        self.norm = RMSNorm(config.d_model)

        # ─── Cabeça de linguagem ───────────────────────────────
        # Projeta d_model → vocab_size, produzindo os logits.
        self.lm_head = BitLinear(config.d_model, config.vocab_size, bias=False)

        # ─── Weight tying ──────────────────────────────────────
        # Faz o lm_head usar a MESMA matriz do embedding.
        # Economiza 8.2M de params na Mini (mais que o modelo inteiro!).
        self.lm_head.weight = self.embed.weight

        # ─── Inicialização dos pesos ───────────────────────────
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """
        Inicialização customizada.
        Embeddings e pesos de BitLinear já são inicializados nos módulos.
        Aqui só cuidamos de casos que eventualmente usem nn.Linear.
        """
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
            input_ids: [batch, seq_len] — índices de tokens
            labels:    [batch, seq_len] — opcional, para calcular loss

        Retorna:
            dict com:
                - "logits": [batch, seq_len, vocab_size]
                - "loss":   escalar (se labels for fornecido)
        """
        batch, seq_len = input_ids.shape

        # ─── 1. Embedding ──────────────────────────────────────
        # [B, S] → [B, S, D]
        x = self.embed(input_ids)

        # ─── 2. Passa pelos blocos ─────────────────────────────
        for block in self.blocks:
            x = block(x)

        # ─── 3. Normalização final ─────────────────────────────
        x = self.norm(x)

        # ─── 4. Cabeça de linguagem ────────────────────────────
        # [B, S, D] → [B, S, vocab_size]
        logits = self.lm_head(x)

        # ─── 5. Monta retorno ──────────────────────────────────
        out = {"logits": logits}

        if labels is not None:
            # Loss de cross-entropy para modelagem de linguagem causal.
            # Shift: o modelo prevê o próximo token, então desloca uma posição.
            # logits[:, :-1] → previsões para as posições 0..S-2
            # labels[:, 1:]  → alvos nas posições 1..S-1
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # Cross-entropy com flatten.
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,  # ignora tokens com label -100 (padding)
            )
            out["loss"] = loss

        return out

    def num_parameters(self, only_trainable: bool = True) -> int:
        """Conta parâmetros. Se only_trainable, ignora buffers."""
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())