"""
Attention — atenção causal com BitLinear e RoPE.

O QUE FAZ:
    Implementa o mecanismo de atenção scaled dot-product com:
      - Projeções Q, K, V usando BitLinear (pesos ternários)
      - RoPE aplicado em Q e K (posição relativa)
      - Máscara causal (cada token só vê os anteriores)

DESENHO:
    entrada x [B, S, D]
        │
        ├─ qkv = BitLinear(D → 3·D)(x)      # uma projeção só, mais eficiente
        ├─ split em Q, K, V [B, S, D] cada
        ├─ reshape → [B, H, S, Dh]           # H = n_heads, Dh = head_dim
        ├─ RoPE em Q e K
        ├─ scores = Q @ K^T / sqrt(Dh)       # [B, H, S, S]
        ├─ mask causal (zera triângulo superior)
        ├─ weights = softmax(scores)
        ├─ out = weights @ V                 # [B, H, S, Dh]
        ├─ reshape → [B, S, D]
        └─ BitLinear(D → D) → saída

POR QUE UMA PROJEÇÃO ÚNICA PARA QKV:
    Fazer 3 BitLinear separadas é mais claro de ler, mas 3 chamadas de
    kernel. Uma única chamada com saída 3× maior é mais eficiente.

POR QUE DIVIDIR POR sqrt(head_dim):
    Sem essa escala, os scores crescem com head_dim, e o softmax satura
    (fica quase one-hot). Dividir mantém a variância controlada.

POR QUE CAUSAL:
    Durante o treino, cada token precisa prever o PRÓXIMO. Se ele visse
    tokens futuros, seria "trapaça". A máscara causal zera o triângulo
    superior da matriz de scores.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from bitlinear import BitLinear
from rope import RoPE

class Attention(nn.Module):
    """
    Atenção causal com BitLinear e RoPE.

    Argumentos:
        d_model:  dimensão do modelo (ex: 256)
        n_heads:  número de cabeças (ex: 4). d_model deve ser divisível por n_heads.
        max_seq_len: comprimento máximo de sequência (para pré-computar RoPE)
        dropout:  taxa de dropout na atenção (0.0 = desligado)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Validação: d_model precisa ser divisível por n_heads
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) precisa ser divisível por "
                f"n_heads ({n_heads})"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len

        # Projeção única para Q, K e V — saída 3× d_model.
        # É mais eficiente que 3 chamadas separadas.
        self.qkv_proj = BitLinear(d_model, 3 * d_model, bias=False)

        # Projeção de saída.
        self.out_proj = BitLinear(d_model, d_model, bias=False)

        # RoPE aplicado em Q e K.
        self.rope = RoPE(head_dim=self.head_dim, max_seq_len=max_seq_len)

        # Dropout na atenção (default 0 — modelo treinado do zero, sem overfit precoce).
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Escala: 1/sqrt(head_dim). Aplicada nos scores antes do softmax.
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Cache da máscara causal (por tamanho de sequência).
        # Evita recriar a matriz toda vez no forward.
        self._causal_mask_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Retorna a máscara causal para a sequência.

        Máscara = tensor [seq_len, seq_len] com:
            - 0 onde pode atender (posição j <= i)
            - -inf onde NÃO pode atender (posição j > i)

        Usa -inf porque, após o softmax, exp(-inf) = 0 — bloqueia completamente.
        """
        key = (seq_len, device)
        if key in self._causal_mask_cache:
            return self._causal_mask_cache[key]

        # Cria diagonal: -inf acima da diagonal principal, 0 no resto.
        mask = torch.full(
            (seq_len, seq_len),
            float("-inf"),
            dtype=torch.float32,
            device=device,
        )
        # torch.triu com diagonal=1 pega o triângulo estritamente superior.
        mask = torch.triu(mask, diagonal=1)

        self._causal_mask_cache[key] = mask
        return mask

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward da atenção.

        Argumentos:
            x:               [batch, seq_len, d_model]
            attention_mask:  opcional [batch, seq_len] — 1 = token válido, 0 = padding.
                             (não usado por enquanto, mas fica pronto)

        Retorna:
            out: [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = x.shape

        # ─── 1. Projeção QKV ─────────────────────────────────────
        # [B, S, D] → [B, S, 3D]
        qkv = self.qkv_proj(x)

        # Divide em Q, K, V cada um com [B, S, D].
        # chunks(3) divide a última dimensão em 3 partes iguais.
        q, k, v = qkv.chunk(3, dim=-1)

        # ─── 2. Reshape para múltiplas cabeças ──────────────────
        # [B, S, D] → [B, S, H, Dh] → [B, H, S, Dh]
        # `.transpose(1, 2)` troca as dimensões 1 (seq) e 2 (heads).
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # ─── 3. RoPE em Q e K ───────────────────────────────────
        # RoPE opera em [B, H, S, Dh] e retorna o mesmo shape.
        q, k = self.rope(q, k)

        # ─── 4. Scaled dot-product attention ────────────────────
        # scores = Q @ K^T / sqrt(Dh)
        # [B, H, S, Dh] @ [B, H, Dh, S] → [B, H, S, S]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # ─── 5. Máscara causal ──────────────────────────────────
        # Soma a máscara: posições bloqueadas recebem -inf.
        # Após softmax, viram 0.
        # IMPORTANTE: cast da máscara para o dtype de x — sem isso, a soma
        # promove para fp32 e o matmul final quebra (fp32 @ bf16).
        causal_mask = self._get_causal_mask(seq_len, x.device).to(x.dtype)
        scores = scores + causal_mask

        # ─── 6. Softmax ─────────────────────────────────────────
        # Softmax na última dimensão (sobre os tokens que pode atender).
        # Em fp32 pra estabilidade numérica (bf16 satura fácil no softmax).
        # Depois, cast explícito para v.dtype — não scores.dtype — para
        # garantir compatibilidade com o matmul seguinte.
        attn_weights = torch.softmax(scores.float(), dim=-1).to(v.dtype)

        # Dropout opcional (não faz nada se dropout=0).
        attn_weights = self.attn_dropout(attn_weights)

        # ─── 7. Aplicar atenção nos values ──────────────────────
        # [B, H, S, S] @ [B, H, S, Dh] → [B, H, S, Dh]
        out = torch.matmul(attn_weights, v)

        # ─── 8. Juntar cabeças ──────────────────────────────────
        # [B, H, S, Dh] → [B, S, H, Dh] → [B, S, D]
        # `.contiguous()` é necessário porque `.transpose` não copia memória.
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)

        # ─── 9. Projeção de saída ───────────────────────────────
        out = self.out_proj(out)

        return out

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"head_dim={self.head_dim}, max_seq_len={self.max_seq_len}"
        )
