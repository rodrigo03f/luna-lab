# Caderno de Laboratório — Luna

Registro cronológico de experimentos, decisões e aprendizados.

Formato de cada entrada:

    ## [DATA] — [TÍTULO CURTO]

    **Objetivo:** o que queríamos descobrir
    **Método:** como fizemos
    **Resultado:** o que medimos
    **Decisão:** aceitar / iterar / descartar
    **Próximo:** o que vem depois

---

## [07/10/2026] — Fundação do laboratório

**Objetivo:** estabelecer protocolo e estrutura antes de codar.
**Método:** conversa + criação de docs.
**Resultado:** LAB_PROTOCOL.md v1.0 criado, estrutura de pastas definida.
**Decisão:** seguir com baseline Transformer como ponto de partida.
**Próximo:** EXP_001 — perfil de VRAM do Transformer baseline.

---

## [07/10/2026 — noite] — A Luna existe

**Objetivo:** implementar as peças da Luna Mini do zero, cada uma testada.
**Método:** implementação incremental, uma peça por vez, teste unitário
antes de ir pra próxima. Tudo em PyTorch puro (Triton vem depois, só
para o que o perfil apontar como gargalo).

**Resultado:**
- 6 peças implementadas: BitLinear, RMSNorm, RoPE, Attention, SwiGLU FFN, LunaForCausalLM
- 30+ testes unitários passando (5/3/5/6/5/8 por peça)
- Luna Mini: 11.6M params, 72 MB VRAM em bf16 na 4060
- Forward, backward, geração: funcionais
- Loss inicial: 10.39 (≈ log(32000) = 10.37, esperado pra modelo não treinado)

**Bug importante encontrado:** RoPE guardava frequências como complex64.
Quando o modelo era castado para bf16, PyTorch descartava silenciosamente
a parte imaginária. RoPE virava "multiplicar por cos" — shape certo,
matemática errada. Corrigido na v2 (cos e sin separados, sem complexos).
Isso gerou a Regra 7 do protocolo.

**Decisão:** seguir para o train.py e treinar a Mini com dados sintéticos.

**Próximo:** train.py (loop, checkpoint, resume, scheduler). Depois
treino end-to-end da Mini. Depois EXP_001 (LOT vs AdamW).

---

## [08/10/2026] — Fase 0 completa: pipeline de dados end-to-end

**Objetivo:** validar o pipeline de dados de ponta a ponta com uma
amostra real antes de comprometer os 50B tokens finais.

**Método:** baixar amostra do GigaVerbo-v2, curar com NeMo Curator,
treinar tokenizer próprio, tokenizar no formato Megatron (.bin/.idx).

**Pipeline executado:**

1. Treinar tokenizer
   - Byte-Level BPE 32k, mistura PT (GigaVerbo) + código (CodeParrot) + inglês (Wikipedia EN)
   - Resultado: 1.3 tokens/palavra em PT, indentação de 4 espaços = 1 token
   - Salvo em: Luna/tokenizer/luna_tokenizer.json

2. Baixar amostra
   - GigaVerbo-v2 em streaming, 200k documentos
   - 1.3 GB de texto bruto

3. Curar com NeMo Curator
   - Filtros: WordCountFilter (10-100k palavras) + NonAlphaNumericFilter (max 20% não-alfanumérico)
   - 972 MB de texto curado (25% de redução)
   - 8 arquivos JSONL

4. Tokenizar com MegatronTokenizerWriter
   - 8 arquivos .bin (uint16) + 8 arquivos .idx
   - 434 MB total
   - 225.9M tokens, 188.9k docs, ratio 1196 tokens/doc

**Bug encontrado:** append_eod=True escreve ID 32000 (inválido) em
0.07% dos tokens. Corrigido usando append_eod=False. Documentação
do bug em Luna/scripts/tokenize_data.py.

**Decisão:** Fase 0 validada. Próximo passo: mini-treino com dados
reais para confirmar que o pipeline de treino lê o formato correto.

**Aprendizados técnicos:**
- JSONL deve ser dividido por LINHAS (split -l), nunca por bytes (split -b)
- NeMo Curator v1.3.0: JsonlReader usa `fields` (lista), não `text_field`
- Ray/Xenna: warning "blocking ray.get" é interno do NeMo, ignorar
- NEMO_CURATOR_LOG_LEVEL e LOGURU_LEVEL não silenciam o Ray — filtrar no shell

**Próximo:** treinar Luna Mini com dados reais e ver loss cair abaixo de 5.

---

## [09/10/2026] — 🌙 Primeira geração de português

**Objetivo:** validar que a Luna Mini aprendeu com dados reais.

**Método:** após 20.000 steps de treino com 226M tokens reais do GigaVerbo
curado, rodar generate.py com 5 prompts diferentes (PT, código, vazio).

**Resultado:**
Loss final: 5.33 (perplexidade 206). Geração produziu texto em português
GRAMATICALMENTE CORRETO, com pontuação e estruturação coerentes.

Exemplos:
- Prompt "A Luna é uma inteligência artificial"
  → "..., mas, não só, de uma alimentação saudável. A maioria das
     pessoas, como se sabe, são mais tarde e os pesquisadores..."

- Prompt "def somar(a, b):"
  → ")\n- Lê-lo\n- Fê..." (reconheceu código como estilo, fechou parêntese)

**Análise:** a Mini aprendeu ESTILO (artigo, notícia, lista, código) sem
compreender SIGNIFICADO. É "papagaio culto" — gramática impecável,
semântica incoerente. Resultado MUITO acima do esperado para 11M params.

**Aprendizado:** a combinação tokenizer de qualidade + curadoria agressiva
+ arquitetura BitLinear produz texto fluente mesmo em escala pequena.

**Decisão:** Fase 0 oficialmente completa. Próximo: EXP_001 (LOT).

---
