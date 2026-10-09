# Protocolo do Laboratório Luna

**Versão:** 1.1
**Data:** 08/10/2026
**Laboratório:** Dr. Rodrigo & Dr. Deep

---

## Filosofia

A Luna não é um Transformer por tradição, nem um Mamba por moda.
A Luna é o resultado de cirurgias guiadas por medição.

Cada peça entra porque passou no teste, não porque é consenso.
Cada peça sai porque outra passou no teste no lugar dela.

O objetivo não é copiar arquitetura. É ter a melhor configuração
possível para a RTX 4060, medida, não achada.

---

## Estrutura do laboratório

    ~/Luna/
    ├── Luna/              # Projeto real. A Luna que vai falar.
    ├── experiments/       # Cada experimento isolado em sua pasta.
    │   ├── EXP_001_vram_baseline/
    │   ├── EXP_002_butterfly_kernel/
    │   └── ...
    ├── data/              # Dados compartilhados
    │   ├── raw/
    │   └── processed/
    ├── docs/              # Protocolo, log, notas de pesquisa
    ├── _legacy/           # Código antigo arquivado (não usar)
    └── README.md

Regra: nada em Luna/ que não passou por um experimento em experiments/.
Nada em experiments/ que não tenha objetivo.md e resultado.md.

---

## As 7 Regras de Ouro

### Regra 1 — Baseline funcional antes de qualquer otimização

Antes de trocar qualquer peça, precisa existir um modelo que treina e
gera texto com peças prontas. Esse é o controle. Sem ele, não há
referência. Sem referência, não há melhoria.

### Regra 2 — Uma peça por vez

Nunca trocar duas coisas no mesmo experimento. Atenção ou otimizador.
Butterfly ou MoE. Um parafuso por vez, para saber o porquê.

### Regra 3 — Benchmark em dois níveis

- Nível 1 (isolado): kernel puro vs. referência PyTorch. VRAM, tempo, corretude.
- Nível 2 (integrado): kernel dentro do modelo real. Tokens/s, loss, VRAM pico.

Se Nível 1 verde e Nível 2 vermelho, o problema é integração, não kernel.

### Regra 4 — Critério de aceitação declarado ANTES

Antes de rodar, escrever: "aceito se >= X% melhor em VRAM ou >= Y%
melhor em tokens/s, sem perda de loss > Z%." Sem isso, aceitamos
melhoria de 2% que custou 3 semanas.

### Regra 5 — Funcionalidade antes de otimização

Sempre ter um modelo que funciona. Otimização é refinamento, nunca
substituição. Se otimizar quebra, volta atrás.

### Regra 6 — Triton só quando o perfil manda

Reescrever em Triton apenas o que o perfil aponta como gargalo.
Exceções iniciais (podem mudar conforme medição):

- SIM: butterfly, BitLinear, MoE router, scan SSM
- TALVEZ: SwiGLU, Attention (só se gargalar)
- NÃO: RoPE, RMSNorm, Embedding, lm_head (custo/benefício ruim)

### Regra 7 — Teste de mixed precision precisa checar corretude numérica

Testes que rodam o modelo em bf16/fp16 devem verificar que a saída
numérica continua correta, não só que a shape está certa.

Bugs de "cast silencioso" (como complex64 -> bf16 descartando a parte
imaginária do RoPE) só aparecem assim. Rodar com shape certo mas
matemática errada é pior que falhar — dá falsa confiança.

---

## Ciclo de vida de um experimento

1. Criar pasta: experiments/EXP_XXX_nome/
2. Escrever objetivo.md ANTES de codar. O que queremos provar?
3. Implementar codigo.py (kernel, módulo, o que for)
4. Escrever benchmark.py (script de medição)
5. Rodar e registrar em resultado.md:
   - Números crus (VRAM, tempo, loss)
   - Comparação com baseline
   - Decisão: aceitar / iterar / descartar
6. Se aceito: integrar em Luna/ e documentar em docs/RESEARCH_LOG.md
7. Se descartado: mantém pasta como referência histórica

---

## Estrutura de cada experimento

    experiments/EXP_XXX_nome/
    ├── objetivo.md      # o que queremos provar
    ├── codigo.py        # implementação
    ├── benchmark.py     # script de medição
    ├── resultado.md     # o que medimos + decisão
    └── artefatos/       # snapshots, gráficos, logs

---

## Convenção de nomes

- Experimentos: EXP_001_nome_curto, EXP_002_outro_nome
- Numeração sequencial, nunca reaproveitada
- Nome em minúsculas, sem acento, separado por underscore

---

## Metas de referência (Luna v1)

| Métrica | Alvo |
|---|---|
| VRAM treino | <= 5.5 GB |
| Tempo 1K steps | <= 2h |
| Params totais | ~500M |
| Params ativos/token | ~120M |
| Inferência | <= 6 GB, chat funcional |

Se um experimento ajuda a chegar nessas metas, ele vale.
Se afasta, ele não vale — por mais bonito que seja o código.

---

## Quando a Luna "está pronta"

Critério de marco: a Luna fala uma frase completa e coerente em
português, e mantém coerência em várias frases seguidas.

Nesse momento, o protocolo de documentação detalhada começa. Antes
disso, o foco é fazer a Luna existir, não documentar cada passo.

---

## O que este protocolo NÃO é

- Não é burocracia. É o que nos impede de virar Frankenstein.
- Não é rígido. Se uma regra atrapalhar, discutimos e revisamos.
- Não é permanente. Versão 2.0 virá quando aprendermos mais.

---

## Registro de revisões

| Versão | Data | Mudança |
|---|---|---|
| 1.0 | 07/10/2026 | Versão inicial |
| 1.1 | 08/10/2026 | Adicionada Regra 7 (mixed precision + corretude numérica) |
