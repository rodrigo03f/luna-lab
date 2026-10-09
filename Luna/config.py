"""
Luna Config — especificação parametrizável da arquitetura.

Este arquivo é a ÚNICA fonte de verdade sobre a arquitetura da Luna.
Nenhum outro arquivo deve ter números mágicos (hardcoded) — tudo vem daqui.

Suporta todas as escalas via arquivos YAML em `configs/`:
    - luna_mini      (~12M params, denso, sem MoE)     — valida pipeline
    - luna_pequena   (~60M params, MoE)                — valida roteamento
    - luna_media     (~180M params, MoE + Butterfly)   — valida escala média
    - luna_grande    (~500M params, alvo final)        — a Luna

Uso típico:
    from luna_model.config import LunaConfig
    cfg = LunaConfig.from_yaml("configs/luna_mini.yaml")
    print(cfg.estimate_params())
"""

# Habilita tipos modernos (list[int], tuple[float, float]) sem quebrar
# em versões antigas do Python.
from __future__ import annotations

# `dataclass` transforma a classe em uma config automática: gera __init__,
# __repr__, __eq__ — tudo de graça.
from dataclasses import asdict, dataclass, field

# `Path` é a forma portável de lidar com caminhos de arquivo (Windows/Linux).
from pathlib import Path

# `Literal` restringe valores. Ex: optimizer só aceita "adamw_fp32" ou "lot".
# Se você digitar "lion", o Python reclama antes de rodar.
from typing import Literal

# `yaml` é o formato dos nossos arquivos de config. Mais legível que JSON,
# e permite comentários dentro dos configs.
import yaml

@dataclass
class LunaConfig:
    """
    Configuração completa da Luna.

    Todos os campos têm defaults que descrevem a **Luna Mini** (a menor).
    Para outras escalas, sobrescreva via YAML.
    """

    # ─────────────────────────────────────────────────────────────
    # IDENTIFICAÇÃO
    # Serve pra rastrear qual config gerou qual checkpoint.
    # ─────────────────────────────────────────────────────────────
    name: str = "luna_mini"       # nome da config (bate com o arquivo YAML)
    version: str = "1.0.0"        # versão do modelo (muda quando arquitetura muda)

    # ─────────────────────────────────────────────────────────────
    # NÚCLEO (Transformer)
    # Esses são os parâmetros que definem o "esqueleto" do modelo.
    # ─────────────────────────────────────────────────────────────
    d_model: int = 256            # dimensão dos embeddings e das ativações
    n_layers: int = 4             # número de blocos (atenção + FFN) empilhados
    n_heads: int = 4              # cabeças de atenção. d_model/n_heads = head_dim
    max_seq_len: int = 512        # comprimento máximo de sequência
    vocab_size: int = 32000       # tamanho do vocabulário (BPE 32k)
    dropout: float = 0.0          # dropout. 0.0 porque treinamos do zero, sem overfit precoce

    # ─────────────────────────────────────────────────────────────
    # FFN / MoE
    # A Luna pode ter FFN denso (Mini) ou FFN esparso via MoE (a partir da Pequena).
    # `use_moe` decide qual dos dois modos é usado.
    # ─────────────────────────────────────────────────────────────
    use_moe: bool = False         # Mini começa densa pra validar o pipeline puro
    d_ff: int = 768               # FFN interno (usado se use_moe=False)
    n_experts: int = 8            # número de experts no MoE (usado se use_moe=True)
    top_k: int = 2                # quantos experts cada token ativa (esparsidade)
    d_ff_expert: int = 1536       # FFN interno de CADA expert (usado se use_moe=True)

    # ─────────────────────────────────────────────────────────────
    # QUANTIZAÇÃO (BitLinear)
    # Pesos ternários (-1, 0, 1) nas camadas de atenção.
    # Ligado desde a Mini porque é a arquitetura da Luna, não uma otimização.
    # ─────────────────────────────────────────────────────────────
    use_bitlinear: bool = True

    # ─────────────────────────────────────────────────────────────
    # BUTTERFLY
    # Comprime os experts do MoE em matrizes estruturadas (menos VRAM por expert).
    # Desligado na Mini e Pequena; ligado a partir da Média.
    # ─────────────────────────────────────────────────────────────
    use_butterfly: bool = False

    # ─────────────────────────────────────────────────────────────
    # MoE — coeficiente da aux loss (Switch Transformer)
    # 0.01 é o padrão do paper. Se os experts colapsarem (todos pro
    # mesmo), aumenta. Se o treino ficar instável, diminui.
    # ─────────────────────────────────────────────────────────────
    moe_aux_loss_coef: float = 0.01

    # ─────────────────────────────────────────────────────────────
    # OTIMIZADOR
    # "lot"         = Luna Ternary Optimizer (o nosso, oficial)
    # "adamw_fp32"  = controle experimental (baseline honesto)
    # ─────────────────────────────────────────────────────────────
    optimizer: Literal["adamw_fp32", "lot"] = "lot"
    lr: float = 3e-4                              # learning rate base
    weight_decay: float = 0.1                     # regularização L2
    betas: tuple[float, float] = (0.9, 0.95)      # momentos do otimizador
    grad_clip: float = 1.0                        # clip de norma do gradiente
    warmup_steps: int = 500                       # steps de aquecimento do LR

    # ─────────────────────────────────────────────────────────────
    # TREINO
    # Parâmetros do loop de treino.
    # ─────────────────────────────────────────────────────────────
    batch_size: int = 8                           # sequências por step
    gradient_accumulation: int = 1                # acumula gradientes antes de atualizar
    max_steps: int = 20000                        # steps totais de treino
    eval_interval: int = 1000                     # a cada N steps, roda eval
    save_interval: int = 1000                     # a cada N steps, salva checkpoint

    # ─────────────────────────────────────────────────────────────
    # PRECISÃO E MEMÓRIA
    # Como o modelo é treinado em termos de dtype e uso de VRAM.
    # ─────────────────────────────────────────────────────────────
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"   # precisão das ativações
    gradient_checkpointing: bool = True               # trade compute por VRAM
        
        # ── Dados ────────────────────────────────────────────────
    # use_real_data=False -> usa dataset sintético (padrão aritmético)
    # use_real_data=True  -> usa .bin/.idx reais do Megatron
    use_real_data: bool = False
    data_dir: str = ""
    

        # ─────────────────────────────────────────────────────────────
    # VALIDAÇÃO
    # __post_init__ é chamado automaticamente depois do __init__.
    # Serve pra checar consistência: se os números não batem, avisa ANTES
    # de tentar treinar (evita descobrir erro no meio do treino).
    # ─────────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        # d_model precisa ser divisível por n_heads.
        # Motivo: cada cabeça usa d_model / n_heads dimensões. Se não for inteiro,
        # o reshape da atenção quebra.
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) precisa ser divisível por "
                f"n_heads ({self.n_heads})."
            )

        # Se MoE está ligado, top_k não pode ser maior que o número de experts.
        # Motivo: não dá pra escolher 4 experts se só existem 2.
        if self.use_moe and self.top_k > self.n_experts:
            raise ValueError(
                f"top_k ({self.top_k}) não pode ser maior que "
                f"n_experts ({self.n_experts})."
            )

        # Butterfly só faz sentido se houver MoE.
        # Motivo: butterfly é uma técnica de compressão de experts.
        if self.use_butterfly and not self.use_moe:
            raise ValueError(
                "use_butterfly=True requer use_moe=True. "
                "Butterfly comprime experts — sem experts, não há o que comprimir."
            )

        # Aviso: se o otimizador for LOT, avisa que é experimental.
        # Motivo: é bom ter isso no log, pra quando a gente olhar atrás saber
        # que aquele checkpoint usou um otimizador em desenvolvimento.
        if self.optimizer == "lot":
            # Não é erro, é aviso. Só garante que a gente lembre.
            pass  # (deixamos o aviso pro train.py imprimir)

    # ─────────────────────────────────────────────────────────────
    # SERIALIZAÇÃO — YAML
    # Métodos de classe pra carregar e salvar configs em YAML.
    # ─────────────────────────────────────────────────────────────
    @classmethod
    def from_yaml(cls, path: str | Path) -> "LunaConfig":
        """
        Carrega uma config de um arquivo YAML.

        Como usar:
            cfg = LunaConfig.from_yaml("configs/luna_mini.yaml")

        O YAML só precisa ter os campos que diferem do default.
        Campos omitidos ficam com o valor padrão do dataclass.
        """
        path = Path(path)

        # Se o arquivo não existe, avisa com clareza.
        if not path.exists():
            raise FileNotFoundError(f"Config não encontrada: {path}")

        # Abre e lê o YAML.
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # `betas` no YAML vem como lista [0.9, 0.95]; o dataclass espera tupla.
        # Convertendo aqui pra manter compatibilidade.
        if "betas" in data and isinstance(data["betas"], list):
            data["betas"] = tuple(data["betas"])

        # Cria a config com os valores lidos. O que não estiver no YAML
        # usa o default do dataclass.
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """
        Salva a config atual em um arquivo YAML.

        Útil pra versionar: cada checkpoint vem com seu config.yaml do lado,
        assim você sabe exatamente qual arquitetura gerou aquele peso.
        """
        path = Path(path)

        # Garante que a pasta existe antes de escrever.
        path.parent.mkdir(parents=True, exist_ok=True)

        # `asdict` transforma o dataclass em dicionário.
        # `tupla → lista` porque YAML não tem tipo tupla nativo.
        data = asdict(self)
        if isinstance(data.get("betas"), tuple):
            data["betas"] = list(data["betas"])

        # Escreve com formatação legível (indentação, ordem preservada).
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    # ─────────────────────────────────────────────────────────────
    # VALIDAÇÃO
    # __post_init__ é chamado automaticamente depois do __init__.
    # Serve pra checar consistência: se os números não batem, avisa ANTES
    # de tentar treinar (evita descobrir erro no meio do treino).
    # ─────────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        # d_model precisa ser divisível por n_heads.
        # Motivo: cada cabeça usa d_model / n_heads dimensões. Se não for inteiro,
        # o reshape da atenção quebra.
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) precisa ser divisível por "
                f"n_heads ({self.n_heads})."
            )

        # Se MoE está ligado, top_k não pode ser maior que o número de experts.
        # Motivo: não dá pra escolher 4 experts se só existem 2.
        if self.use_moe and self.top_k > self.n_experts:
            raise ValueError(
                f"top_k ({self.top_k}) não pode ser maior que "
                f"n_experts ({self.n_experts})."
            )

        # Butterfly só faz sentido se houver MoE.
        # Motivo: butterfly é uma técnica de compressão de experts.
        if self.use_butterfly and not self.use_moe:
            raise ValueError(
                "use_butterfly=True requer use_moe=True. "
                "Butterfly comprime experts — sem experts, não há o que comprimir."
            )

        # Aviso: se o otimizador for LOT, avisa que é experimental.
        # Motivo: é bom ter isso no log, pra quando a gente olhar atrás saber
        # que aquele checkpoint usou um otimizador em desenvolvimento.
        if self.optimizer == "lot":
            # Não é erro, é aviso. Só garante que a gente lembre.
            pass  # (deixamos o aviso pro train.py imprimir)

    # ─────────────────────────────────────────────────────────────
    # SERIALIZAÇÃO — YAML
    # Métodos de classe pra carregar e salvar configs em YAML.
    # ─────────────────────────────────────────────────────────────
    @classmethod
    def from_yaml(cls, path: str | Path) -> "LunaConfig":
        """
        Carrega uma config de um arquivo YAML.

        Como usar:
            cfg = LunaConfig.from_yaml("configs/luna_mini.yaml")

        O YAML só precisa ter os campos que diferem do default.
        Campos omitidos ficam com o valor padrão do dataclass.
        """
        path = Path(path)

        # Se o arquivo não existe, avisa com clareza.
        if not path.exists():
            raise FileNotFoundError(f"Config não encontrada: {path}")

        # Abre e lê o YAML.
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # `betas` no YAML vem como lista [0.9, 0.95]; o dataclass espera tupla.
        # Convertendo aqui pra manter compatibilidade.
        if "betas" in data and isinstance(data["betas"], list):
            data["betas"] = tuple(data["betas"])

        # Cria a config com os valores lidos. O que não estiver no YAML
        # usa o default do dataclass.
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """
        Salva a config atual em um arquivo YAML.

        Útil pra versionar: cada checkpoint vem com seu config.yaml do lado,
        assim você sabe exatamente qual arquitetura gerou aquele peso.
        """
        path = Path(path)

        # Garante que a pasta existe antes de escrever.
        path.parent.mkdir(parents=True, exist_ok=True)

        # `asdict` transforma o dataclass em dicionário.
        # `tupla → lista` porque YAML não tem tipo tupla nativo.
        data = asdict(self)
        if isinstance(data.get("betas"), tuple):
            data["betas"] = list(data["betas"])

        # Escreve com formatação legível (indentação, ordem preservada).
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


    # ─────────────────────────────────────────────────────────────
    # ESTIMATIVAS
    # Calcula params totais e VRAM estimada SEM construir o modelo.
    # Útil pra decidir config ANTES de gastar tempo treinando.
    # ─────────────────────────────────────────────────────────────
    def estimate_params(self) -> dict[str, int]:
        """
        Estima o número de parâmetros por componente.

        Retorna um dicionário com:
            - embedding       : tabela de embeddings
            - attention       : projeções QKV + out
            - ffn             : FFN denso ou MoE (todos os experts)
            - ffn_active      : parte do FFN ativada por token (MoE só ativa top_k)
            - lm_head         : projeção final (pode ser 0 se weight tying)
            - total           : soma de tudo
            - total_active    : params ativos por token (relevante pra MoE)
        """
        d = self.d_model
        V = self.vocab_size

        # Embedding: uma matriz [vocab_size, d_model]
        embedding = V * d

        # Atenção (por camada): 4 projeções [d_model, d_model] (Q, K, V, O)
        # Multiplicado por n_layers.
        attn_per_layer = 4 * d * d
        attention = attn_per_layer * self.n_layers

        # FFN
        if self.use_moe:
            # SwiGLU usa 3 matrizes por expert: gate [d, d_ff], up [d, d_ff], down [d_ff, d]
            # Total = n_experts × 3 × d × d_ff_expert
            ffn_total = self.n_experts * 3 * d * self.d_ff_expert

            # Por token, só top_k experts são ativados.
            ffn_active = self.top_k * 3 * d * self.d_ff_expert
        else:
            # FFN denso: 2 matrizes [d, d_ff] + [d_ff, d]
            ffn_total = 2 * d * self.d_ff
            ffn_active = ffn_total

        ffn_total *= self.n_layers
        ffn_active *= self.n_layers

        # LM head: matriz [d_model, vocab_size]
        # NOTA: na Luna usamos weight tying (embedding ↔ lm_head),
        # então essa matriz NÃO é contada separado (é a mesma do embedding).
        lm_head = 0

        total = embedding + attention + ffn_total + lm_head
        total_active = embedding + attention + ffn_active + lm_head

        return {
            "embedding": embedding,
            "attention": attention,
            "ffn": ffn_total,
            "ffn_active": ffn_active,
            "lm_head": lm_head,
            "total": total,
            "total_active": total_active,
        }

    def estimate_vram_mb(self, optimizer: str | None = None) -> dict[str, float]:
        """
        Estima VRAM em MB por componente, para o otimizador escolhido.

        Assume pesos e gradientes em bf16 (2 bytes), ativações estimadas
        de forma aproximada, e o estado do otimizador variando conforme a escolha.

        Parâmetro `optimizer` sobrescreve o da config (útil pra simular).
        Retorna um dicionário com pesos, gradientes, otimizador, ativações e total.
        """
        optimizer = optimizer or self.optimizer
        params = self.estimate_params()
        total_params = params["total"]

        # Pesos em bf16: 2 bytes por parâmetro
        weights_mb = total_params * 2 / 1024 / 1024

        # Gradientes em bf16: mesma conta dos pesos
        grads_mb = total_params * 2 / 1024 / 1024

        # Estado do otimizador — depende do escolhido
        if optimizer == "adamw_fp32":
            # AdamW fp32 guarda m1 + m2 em fp32 (4 bytes cada) = 8 bytes/param
            optim_mb = total_params * 8 / 1024 / 1024
        elif optimizer == "lot":
            # LOT (Luna Ternary Optimizer): estado ternário, ~0.25 bytes/param
            # (2 bits por parâmetro)
            optim_mb = total_params * 0.25 / 1024 / 1024
        else:
            raise ValueError(f"Otimizador desconhecido: {optimizer}")

        # Ativações: estimativa grosseira.
        # Fórmula: batch × seq × d_model × n_layers × bytes × fator_checkpoint
        # Fator ~0.3 quando gradient checkpointing está ligado (guarda só entrada de cada bloco)
        seq = self.max_seq_len
        d = self.d_model
        batch = self.batch_size
        checkpoint_factor = 0.3 if self.gradient_checkpointing else 1.0
        acts_mb = (
            batch * seq * d * self.n_layers * 2  # bytes
            * checkpoint_factor
            / 1024 / 1024
        )

        # Overhead CUDA/Triton (constante prática)
        overhead_mb = 300.0

        total_mb = weights_mb + grads_mb + optim_mb + acts_mb + overhead_mb

        return {
            "weights_mb": round(weights_mb, 1),
            "grads_mb": round(grads_mb, 1),
            "optim_mb": round(optim_mb, 1),
            "activations_mb": round(acts_mb, 1),
            "overhead_mb": overhead_mb,
            "total_mb": round(total_mb, 1),
        }

    def summary(self) -> str:
        """
        Retorna um resumo legível da config + estimativas.
        Útil pra imprimir no começo do treino.
        """
        p = self.estimate_params()
        v = self.estimate_vram_mb()

        lines = [
            f"╭─ LunaConfig: {self.name} (v{self.version}) ─────────────",
            f"│  d_model={self.d_model}  n_layers={self.n_layers}  n_heads={self.n_heads}",
            f"│  max_seq_len={self.max_seq_len}  vocab_size={self.vocab_size}",
            f"│  MoE={'on' if self.use_moe else 'off'}"
            + (f" (experts={self.n_experts}, top_k={self.top_k})" if self.use_moe else ""),
            f"│  BitLinear={'on' if self.use_bitlinear else 'off'}  "
            f"Butterfly={'on' if self.use_butterfly else 'off'}",
            f"│  otimizador={self.optimizer}  lr={self.lr}  dtype={self.dtype}",
            f"├─ Parâmetros ────────────────────────────────────────────",
            f"│  total:  {p['total'] / 1e6:.1f} M",
            f"│  ativo:  {p['total_active'] / 1e6:.1f} M",
            f"├─ VRAM estimada (MB) ───────────────────────────────────",
            f"│  pesos:      {v['weights_mb']:.1f}",
            f"│  gradientes: {v['grads_mb']:.1f}",
            f"│  otimizador: {v['optim_mb']:.1f}",
            f"│  ativações:  {v['activations_mb']:.1f}",
            f"│  overhead:   {v['overhead_mb']:.1f}",
            f"│  TOTAL:      {v['total_mb']:.1f}  (teto treino: 5500 MB)",
            f"╰─────────────────────────────────────────────────────────",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# Execução direta: `python config.py`
# Mostra a config default (Mini) e as estimativas.
# Serve pra testar rapidamente antes de partir pro modelo.
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = LunaConfig()
    print(cfg.summary())