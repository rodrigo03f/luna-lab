# Luna v1 — Especificação Arquitetural

**Data:** 07/10/2026
**Status:** Fechada, aguardando implementação
**Hardware alvo:** RTX 4060 8GB

---

## 1. Objetivo

IA pessoal, local, em português + código. Conversa e assistência de
programação. Uso individual.

Tese técnica: **muitos parâmetros totais com pouquíssima VRAM em treino
e inferência** via BitNet b1.58 + MoE esparso + experts em butterfly
+ otimizador ternário.

---

## 2. Restrições de hardware e ambiente

| Item | Valor |
|---|---|
| GPU | RTX 4060, 8 GB |
| Driver / CUDA | 616.92 / CUDA UMD 13.4 |
| VRAM treino disponível | **5.5 GB** |
| Python | 3.12.3 |
| Ambiente | Container `nvcr.io/nvidia/nemo:26.06`, WSL2 |
| Projeto | `~/Luna` (ext4 nativo do WSL2) |
| Pacotes | `uv 0.7.2`, venv em `/opt/venv` |

**Decisões herdadas:**
- NeMo entra apenas como *dados + curadoria + export*, nunca no loop de treino.
- TE / Megatron Core **fora**.
- Tudo em **Triton de verdade**.

---

## 3. Arquitetura

### 3.1 Núcleo

| Parâmetro | Valor |
|---|---|
| Tipo | Decoder-only Transformer |
| `d_model` | 768 |
| `n_layers` | 12 |
| `n_heads` | 12 (head_dim = 64) |
| `max_seq_len` | 2048 |
| `vocab_size` | 32000 |
| Params totais | ~500M |
| Params ativos/token | ~120M |

### 3.2 Componentes

- **Atenção:** `BitLinear` (BitNet b1.58) em `qkv` e `out`.
- **FFN:** MoE com 16 experts, `top_k=2`.
  - Cada expert: W1/W2 em butterfly fixo, log2(d_ff) fatores.
  - `d_ff` interno ~3072 (comprimido).
  - Roteador próprio em Triton.
- **RMSNorm**, **RoPE**, **SwiGLU**, **weight tying**.

### 3.3 Fora do escopo v1

- DeBut (deformable butterfly)
- `max_seq_len=4096`
- Export NeMo AutoModel

---

## 4. Orçamento de VRAM
VRAM total: 8188 MB
(-) Sistema + WSL2: ~1105 MB
(-) LDPlayer (opcional): ~800 MB
(-) Margem de segurança: ~500 MB
─────────────────────────────────────
= Teto treino: 5500 MB

text

---

## 5. Otimizador

| Variante | Estado em VRAM | Status |
|---|---|---|
| AdamW fp32 | ~4 GB | Inviável |
| AdamW 8-bit | ~1 GB | Baseline |
| Lion fp32 | ~2 GB | Referência |
| Lion int8 | ~500 MB | Meio caminho |
| **Lion ternário** | **~125 MB** | **Alvo** |
| GaLore + AdamW | ~500 MB | Plano B |

### Ordem de benchmark

1. AdamW 8-bit
2. Lion-Triton fp32
3. Lion-Triton int8
4. Lion-Triton ternário
5. GaLore + AdamW

**Critério:** melhor razão (tokens/s) / (VRAM pico), loss comparável ao baseline.

---

## 6. Ordem de implementação

Cada etapa termina com benchmark + teste unitário.

1. Butterfly Triton (fwd + bwd)
2. BitLinear Triton (fwd + STE bwd)
3. MoE Triton (router, dispatch, combine)
4. Modelo completo (batch pequeno, checar VRAM)
5. Otimizadores (5 variantes, comparativo)
6. Loop de treino (resume + checkpoints)

**Regra:** nenhuma etapa começa antes da anterior fechar verde.

---

## 7. Métricas de sucesso v1

| Métrica | Alvo |
|---|---|
| VRAM pico treino | ≤ 5.5 GB |
| Tempo 1K steps | ≤ 2h |
| Params totais | 500M |
| Params ativos/token | ~120M |
| Loss descendo | Sim |
| Inferência | ≤ 6 GB, chat funcional |
