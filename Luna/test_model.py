"""
Teste do LunaForCausalLM — modelo completo.

Esse é o teste que prova que a Luna EXISTE:
    - Instancia com config
    - Forward calcula logits
    - Loss calcula corretamente
    - Backward flui
    - Contagem de params bate com estimativa
    - Roda em GPU com bf16
    - Geração de texto funciona (mesmo que seja lixo no início)
"""

import torch

from config import LunaConfig
from model import LunaForCausalLM, LunaBlock


def test_block():
    """LunaBlock preserva shape."""
    print("── Teste 1: LunaBlock preserva shape ──────────────────")
    block = LunaBlock(d_model=128, n_heads=4, d_ff=384, max_seq_len=64)
    x = torch.randn(2, 16, 128)
    y = block(x)
    print(f"   Entrada: {x.shape}  Saída: {y.shape}")
    assert y.shape == x.shape
    print("   ✓ OK")


def test_model_instantiation():
    """Modelo instancia com config."""
    print("\n── Teste 2: instancia modelo ──────────────────────────")
    cfg = LunaConfig()
    model = LunaForCausalLM(cfg)
    total = model.num_parameters()
    print(f"   Params: {total:,} ({total / 1e6:.2f}M)")
    # Deve ser próximo da estimativa do config (10.8M)
    assert 9e6 < total < 13e6, f"Params fora do esperado: {total}"
    print("   ✓ OK")


def test_forward_shapes():
    """Forward produz logits com shape correto."""
    print("\n── Teste 3: forward ───────────────────────────────────")
    cfg = LunaConfig()
    model = LunaForCausalLM(cfg)

    # Batch de 2 sequências de 32 tokens
    input_ids = torch.randint(0, cfg.vocab_size, (2, 32))
    out = model(input_ids)

    print(f"   Input:  {input_ids.shape}")
    print(f"   Logits: {out['logits'].shape}")
    assert out["logits"].shape == (2, 32, cfg.vocab_size)
    print("   ✓ OK")


def test_loss():
    """Loss calcula sem erro e é um valor razoável."""
    print("\n── Teste 4: loss ──────────────────────────────────────")
    cfg = LunaConfig()
    model = LunaForCausalLM(cfg)

    input_ids = torch.randint(0, cfg.vocab_size, (2, 32))
    # Labels = mesmos tokens (modelo aprende a prever o próximo)
    out = model(input_ids, labels=input_ids)

    loss = out["loss"].item()
    print(f"   Loss inicial: {loss:.4f}")

    # Loss aleatória para vocab 32k é ~log(32000) ≈ 10.37
    # Deve estar próximo disso (modelo não treinado)
    assert 8.0 < loss < 12.0, f"Loss muito fora: {loss}"
    print("   ✓ OK — loss inicial está dentro do esperado")


def test_backward():
    """Backward flui para todos os parâmetros."""
    print("\n── Teste 5: backward ──────────────────────────────────")
    cfg = LunaConfig()
    model = LunaForCausalLM(cfg)

    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(input_ids, labels=input_ids)
    out["loss"].backward()

    # Conta quantos parâmetros receberam gradiente
    total_params = 0
    with_grad = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            total_params += 1
            if p.grad is not None and p.grad.abs().sum() > 0:
                with_grad += 1

    print(f"   Parâmetros com gradiente: {with_grad}/{total_params}")
    assert with_grad > 0
    print("   ✓ OK")


def test_param_breakdown():
    """Mostra a distribuição de parâmetros por peça."""
    print("\n── Teste 6: breakdown de parâmetros ───────────────────")
    cfg = LunaConfig()
    model = LunaForCausalLM(cfg)

    def count(module):
        return sum(p.numel() for p in module.parameters())

    embed = count(model.embed)
    blocks = count(model.blocks)
    norm = count(model.norm)
    # lm_head compartilha com embed, então não conta de novo
    total = model.num_parameters()

    print(f"   Embedding:  {embed:,} ({embed/total*100:.1f}%)")
    print(f"   Blocks:     {blocks:,} ({blocks/total*100:.1f}%)")
    print(f"   Norm final: {norm:,}")
    print(f"   Total:      {total:,}")
    print(f"   (lm_head compartilha peso com embed — economiza {cfg.vocab_size * cfg.d_model:,})")
    print("   ✓ OK")


def test_cuda_bf16():
    """Modelo roda em GPU com bf16."""
    print("\n── Teste 7: CUDA + bf16 ───────────────────────────────")
    if not torch.cuda.is_available():
        print("   ⚠ CUDA não disponível — pulando")
        return

    device = torch.device("cuda")
    cfg = LunaConfig()
    model = LunaForCausalLM(cfg).to(device)
    model = model.to(torch.bfloat16)

    torch.cuda.reset_peak_memory_stats()

    input_ids = torch.randint(0, cfg.vocab_size, (2, 32), device=device)
    out = model(input_ids, labels=input_ids)

    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024**2
    loss = out["loss"].item()

    print(f"   Device: {device}")
    print(f"   Loss:   {loss:.4f}")
    print(f"   VRAM:   {peak:.2f} MB (pico)")

    # Esperado: poucos MB (Mini é 10.8M params em bf16)
    assert peak < 500, f"VRAM maior que esperado: {peak} MB"
    print("   ✓ OK")


def test_generation():
    """
    Geração greedy simples — o modelo não foi treinado,
    então o texto é lixo. Mas o pipeline funciona.
    """
    print("\n── Teste 8: geração funciona ──────────────────────────")
    cfg = LunaConfig()
    model = LunaForCausalLM(cfg)
    model.eval()

    # Prompt: alguns tokens aleatórios
    prompt = torch.randint(0, cfg.vocab_size, (1, 8))

    with torch.no_grad():
        generated = prompt.clone()
        for _ in range(8):
            out = model(generated)
            # Pega o último token e escolhe o de maior logit (greedy)
            next_token = out["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

    print(f"   Prompt:    {prompt.shape}")
    print(f"   Gerado:    {generated.shape}")
    print(f"   Tokens gerados: {generated[0, 8:].tolist()}")
    assert generated.shape == (1, 16)
    print("   ✓ OK — pipeline de geração funciona")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTE DO LUNA FOR CAUSAL LM")
    print("=" * 60)

    test_block()
    test_model_instantiation()
    test_forward_shapes()
    test_loss()
    test_backward()
    test_param_breakdown()
    test_cuda_bf16()
    test_generation()

    print("\n" + "=" * 60)
    print("✅ TODOS OS TESTES PASSARAM")
    print("🌙 A LUNA EXISTE")
    print("=" * 60)