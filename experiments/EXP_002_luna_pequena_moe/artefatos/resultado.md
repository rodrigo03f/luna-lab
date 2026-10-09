# EXP_002 — Luna Pequena com MoE

**Data:** 09/10/2026
**Status:** APROVADO

## Objetivo

Validar MoE pela primeira vez em escala real:
- Roteamento (top-k)
- Aux loss e balanceamento
- Integração com BitLinear
- Treino end-to-end

## Config

| Item | Valor |
|---|---|
| Params totais | 60.46M |
| Params ativos | 32.1M |
| d_model | 512 |
| n_layers | 6 |
| n_heads | 8 |
| max_seq_len | 1024 |
| Experts | 8 |
| top_k | 2 |
| d_ff_expert | 512 |
| BitLinear | ON |
| Butterfly | OFF |

## Testes unitários

7/7 passaram. Aux loss medida em ~2.0 (range teórico [1.0, 8.0]),
indicando distribuição quase uniforme entre experts (zero colapso).

## Treino

| Métrica | Valor |
|---|---|
| Steps | 20.000 |
| Tempo | 98 min |
| Loss inicial | 10.43 |
| Loss final | 4.29 |
| Perplexidade | 73 |
| VRAM pico | ~600 MB (estimado) |

## Comparação com Mini

| Modelo | Params | Loss | Perplexidade |
|---|---|---|---|
| Mini | 11.6M | 5.33 | 206 |
| Pequena | 60.5M | 4.29 | 73 |
| Ganho | 5.2x params | -1.04 loss | 2.8x melhor |

## Análise qualitativa da geração

**Mini:** palavras soltas, fragmentos sem coesão.
**Pequena:** frases completas, gramática correta, tom jornalístico.

Exemplo (prompt "A Luna é uma inteligência artificial"):
> "...que é uma técnica que ajuda crianças e adolescentes a saberem
> a gente. A ideia de como um estudante pode ajudar a desenvolver
> novas vidas. No mundo, a produção de novos sonhos é muito
> importante, mas é um grande desafio para todos. Confira abaixo."

Frases individuais corretas, mas parágrafo sem tese única.

## Decisão

EXP_002 APROVADO. MoE funciona. Segue para EXP_003 (Luna Média + Butterfly).