# Laboratório Luna

Repositório do laboratório. Aqui vive a Luna (a IA pessoal em
português + código para a RTX 4060) e todos os experimentos
que a constroem.

## Como navegar

- Luna/          → o projeto real. A Luna que vai falar.
- experiments/   → cada teste isolado, com objetivo e resultado.
- docs/          → protocolo, caderno de laboratório, notas.
- _legacy/       → código antigo, arquivado. Não usar.

## Protocolo em uma linha

    Cirurgia guiada por medição. Uma peça por vez. Baseline antes
    de otimização. Funcionalidade antes de performance.

## Status atual

- [x] Ambiente configurado (container NeMo 26.06, WSL2, GPU visível)
- [x] Protocolo v1.0 escrito
- [ ] EXP_001: perfil de VRAM do Transformer baseline

## Hardware alvo

RTX 4060 8GB | VRAM treino <= 5.5 GB | Inferência <= 6 GB

## Entrar no ambiente

No WSL2, fora do container:

    luna
