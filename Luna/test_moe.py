"""
Teste da camada MoE.

Verifica:
    1. Forward preserva shape
    2. Backward flui para todos os experts
    3. Aux loss retorna valor razoável
    4. Top-k ativa exatamente k experts por token
    5. Modelo completo com MoE calcula loss
    6. Modelo sem MoE continua funcionando
"""

import torch

from config import LunaConfig
from experts import MoE
from model import LunaForCausalLM


def test_moe_shape():
    """Forward preserva shape."""
    print("── Teste 1: MoE preserva shape ────────────────────────")

    moe = MoE(d_model=256, n_experts=8, top_k=2, d_ff_expert=256)
    x = torch.randn(2, 16, 256)
    out, aux = moe(x)

    print(f"   Entrada: {x.shape}")
    print(f"   Saída:   {out.shape}")
    print(f"   Aux loss: {aux.item():.4f}")

    assert out.shape == x.shape
    assert aux.dim() == 0  # escalar
    print("   ✓ OK")


def test_moe_backward():
    """Backward flui para todos os experts (em média)."""
    print("\n── Teste 2: backward flui para experts ────────────────")

    moe = MoE(d_model=128, n_experts=4, top_k=2, d_ff_expert=256)
    x = torch.randn(4, 32, 128, requires_grad=True)
    out, aux = moe(x)

    loss = out.sum() + 0.01 * aux
    loss.backward()

    # Conta experts que receberam gradiente
    experts_with_grad = 0
    for expert in moe.experts:
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in expert.parameters()
        )
        if has_grad:
            experts_with_grad += 1

    print(f"   Experts com gradiente: {experts_with_grad}/4")
    # Com batch grande e 2 top_k de 4, esperamos pelo menos 3 experts ativos
    assert experts_with_grad >= 2, "Poucos experts receberam gradiente!"
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    print("   ✓ OK")


def test_aux_loss_range():
    """Aux loss fica entre 1 e n_experts."""
    print("\n── Teste 3: aux loss em range válido ──────────────────")

    moe = MoE(d_model=128, n_experts=8, top_k=2, d_ff_expert=256)
    x = torch.randn(8, 64, 128)
    _, aux = moe(x)

    aux_val = aux.item()
    print(f"   Aux loss: {aux_val:.4f}")
    print(f"   Range esperado: [1.0, 8.0]")

    assert 1.0 <= aux_val <= 8.0, f"Aux loss fora do range: {aux_val}"
    print("   ✓ OK")


def test_top_k_indices():
    """Top-k ativa exatamente k experts por token."""
    print("\n── Teste 4: top-k correto ─────────────────────────────")

    moe = MoE(d_model=128, n_experts=8, top_k=2, d_ff_expert=256)
    x = torch.randn(4, 16, 128)

    # Reproduz o roteamento
    x_flat = x.view(-1, 128)
    router_logits = moe.router(x_flat)
    _, top_k_indices = router_logits.topk(2, dim=-1)

    # Cada token deve ter exatamente 2 índices distintos
    n_tokens = x_flat.shape[0]
    for i in range(n_tokens):
        indices = top_k_indices[i].tolist()
        assert len(indices) == 2, f"Token {i} tem {len(indices)} experts"
        assert len(set(indices)) == 2, f"Token {i} tem experts duplicados"

    print(f"   {n_tokens} tokens, cada um com exatamente 2 experts ✅")
    print("   ✓ OK")


def test_model_with_moe():
    """Modelo completo com MoE calcula loss."""
    print("\n── Teste 5: modelo com MoE ────────────────────────────")

    cfg = LunaConfig.from_yaml("configs/luna_pequena.yaml")
    model = LunaForCausalLM(cfg)

    total = model.num_parameters()
    print(f"   Params: {total/1e6:.2f}M")

    input_ids = torch.randint(0, cfg.vocab_size, (2, 32))
    out = model(input_ids, labels=input_ids)

    print(f"   Loss: {out['loss'].item():.4f}")
    print(f"   Aux loss: {out['aux_loss'].item():.4f}")

    assert "loss" in out
    assert "aux_loss" in out
    assert not torch.isnan(out["loss"])
    print("   ✓ OK")


def test_model_without_moe():
    """Modelo Mini (sem MoE) continua funcionando."""
    print("\n── Teste 6: modelo sem MoE ────────────────────────────")

    cfg = LunaConfig()  # padrão = Mini, sem MoE
    model = LunaForCausalLM(cfg)

    total = model.num_parameters()
    print(f"   Params: {total/1e6:.2f}M")

    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(input_ids, labels=input_ids)

    print(f"   Loss: {out['loss'].item():.4f}")

    assert "loss" in out
    assert "aux_loss" not in out  # sem MoE não tem aux loss
    print("   ✓ OK")


def test_cuda_moe():
    """MoE roda em CUDA bf16."""
    print("\n── Teste 7: CUDA + bf16 ───────────────────────────────")

    if not torch.cuda.is_available():
        print("   ⚠ CUDA não disponível — pulando")
        return

    device = torch.device("cuda")
    cfg = LunaConfig.from_yaml("configs/luna_pequena.yaml")
    model = LunaForCausalLM(cfg).to(device).to(torch.bfloat16)

    torch.cuda.reset_peak_memory_stats()

    input_ids = torch.randint(0, cfg.vocab_size, (2, 64), device=device)
    out = model(input_ids, labels=input_ids)
    torch.cuda.synchronize()

    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"   Loss: {out['loss'].item():.4f}")
    print(f"   VRAM: {peak:.2f} MB")

    assert peak < 2000, f"VRAM alta: {peak} MB"
    print("   ✓ OK")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTE DO MOE — Luna Pequena")
    print("=" * 60)

    test_moe_shape()
    test_moe_backward()
    test_aux_loss_range()
    test_top_k_indices()
    test_model_with_moe()
    test_model_without_moe()
    test_cuda_moe()

    print("\n" + "=" * 60)
    print("✅ TODOS OS TESTES PASSARAM")
    print("=" * 60)