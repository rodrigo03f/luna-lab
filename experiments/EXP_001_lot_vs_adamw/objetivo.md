# EXP_001 — LOT vs AdamW

**Data de início:** 09/10/2026
**Status:** em andamento

---

## Pergunta

O LOT (Luna Ternary Optimizer) consegue treinar a Luna Mini
comparável ao AdamW fp32, usando muito menos VRAM?

---

## Critério de aceitação (DECLARADO ANTES — Regra 4)

Aceito o LOT como otimizador oficial da Luna se, após 2.000 steps
de treino na Mini com os mesmos dados:

| Métrica | Critério |
|---|---|
| Loss final | <= 5% pior que AdamW fp32 |
| VRAM do otimizador | >= 5x menor que AdamW fp32 |
| Velocidade (st/s) | >= 80% do AdamW fp32 |

Se falhar em qualquer um:
- Loss pior que 5%: ajustar hiperparâmetros, testar variantes
- VRAM não 5x menor: rever design do estado ternário
- Velocidade menor que 80%: otimizar kernel Triton

---

## Hipótese

O LOT deve atingir loss comparável ao AdamW porque:
1. Lion (regra de sinal) já foi validado em escala por Google
2. TACO (estado ternário) já foi validado em 13B
3. A combinação é nova, mas as peças foram testadas

A VRAM deve cair drasticamente porque:
- AdamW fp32: m1 (4B) + m2 (4B) + master (4B) = 12 bytes/param
- LOT ternário: m1 (0.25B) = 0.25 bytes/param
- Redução esperada: ~48x

---

## Metodologia

### Nível 1 — Isolado (kernel)
- Kernel do LOT vs torch.optim.AdamW
- Medir: correção (updates iguais?), tempo de step, VRAM
- Dados sintéticos

### Nível 2 — Integrado (modelo)
- Mesmo modelo (Luna Mini), mesmos dados (226M tokens), mesma seed
- Treinar 2.000 steps com cada otimizador
- Medir: loss, VRAM, st/s, tempo

---

## Variantes a testar (Regra 2 — uma peça por vez)

| # | Otimizador | Estado |
|---|---|---|
| V0 | AdamW fp32 | Baseline (12 bytes/param) |
| V1 | LOT PyTorch (referência) | Corretude primeiro |
| V2 | LOT Triton | Performance |
| V3 | LOT int8 (meio caminho) | Se ternário falhar |

Ordem: V0 -> V1 -> V2 (V3 só se V2 falhar).

---

## Hipóteses alternativas (se V1 falhar)

Se o LOT ternário não convergir:
1. Momentum com escala adaptativa (por tensor, não por parâmetro)
2. LOT + variância em int8 (Adam-like, mais estável)
3. LOT ternário só no MoE (onde o ganho importa mais)

---

## Resultado

(preenchido no final do experimento)
