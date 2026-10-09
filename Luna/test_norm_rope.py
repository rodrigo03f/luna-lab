"""
Teste do RMSNorm e RoPE (v2 — sem complexos).
"""

import torch
from norm import RMSNorm
from rope import RoPE, precompute_freqs


def test_rmsnorm_shape():
    print("── Teste 1: RMSNorm preserva shape ────────────────────")
    norm = RMSNorm(d_model=256)
    x = torch.randn(4, 16, 256)
    y = norm(x)
    print(f"   Entrada: {x.shape}  Saída: {y.shape}")
    assert y.shape == x.shape
    print("   ✓ OK")


def test_rmsnorm_normalizes():
    print("\n── Teste 2: RMSNorm normaliza ─────────────────────────")
    norm = RMSNorm(d_model=128)
    x = torch.randn(2, 8, 128) * 100.0
    y = norm(x)
    rms = torch.sqrt(y.float().pow(2).mean(dim=-1))
    print(f"   RMS médio antes: {torch.sqrt(x.pow(2).mean(dim=-1)).mean().item():.2f}")
    print(f"   RMS médio depois: {rms.mean().item():.4f}")
    assert 0.9 < rms.mean().item() < 1.1
    print("   ✓ OK")


def test_rmsnorm_backward():
    print("\n── Teste 3: RMSNorm backward ──────────────────────────")
    norm = RMSNorm(d_model=64)
    x = torch.randn(2, 4, 64, requires_grad=True)
    y = norm(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    print(f"   Norma do gradiente: {x.grad.norm().item():.4f}")
    print("   ✓ OK")


def test_rope_freqs():
    print("\n── Teste 4: precompute_freqs ──────────────────────────")
    cos, sin = precompute_freqs(head_dim=64, max_seq_len=512)
    print(f"   cos shape: {cos.shape}  dtype: {cos.dtype}")
    print(f"   sin shape: {sin.shape}  dtype: {sin.dtype}")
    assert cos.shape == (512, 32)
    assert sin.shape == (512, 32)
    assert cos.dtype == torch.float32
    assert sin.dtype == torch.float32
    print("   ✓ OK")


def test_rope_shape():
    print("\n── Teste 5: RoPE preserva shape ───────────────────────")
    rope = RoPE(head_dim=64, max_seq_len=512)
    q = torch.randn(2, 4, 32, 64)
    k = torch.randn(2, 4, 32, 64)
    q_rot, k_rot = rope(q, k)
    print(f"   Q: {q.shape} → {q_rot.shape}")
    print(f"   K: {k.shape} → {k_rot.shape}")
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    print("   ✓ OK")


def test_rope_preserves_norm():
    print("\n── Teste 6: RoPE preserva norma ───────────────────────")
    rope = RoPE(head_dim=64, max_seq_len=512)
    q = torch.randn(1, 2, 16, 64)
    k = torch.randn(1, 2, 16, 64)
    q_rot, k_rot = rope(q, k)
    norm_q_antes = q.norm(dim=-1).mean().item()
    norm_q_depois = q_rot.norm(dim=-1).mean().item()
    print(f"   Norma Q antes:  {norm_q_antes:.4f}")
    print(f"   Norma Q depois: {norm_q_depois:.4f}")
    assert abs(norm_q_antes - norm_q_depois) < 0.01
    print("   ✓ OK")


def test_rope_bf16_safe():
    """
    NOVO TESTE CRÍTICO (Regra 7 do protocolo):
    RoPE deve funcionar corretamente em bf16, sem perder informação.
    """
    print("\n── Teste 7: RoPE em bf16 é seguro ─────────────────────")
    rope = RoPE(head_dim=32, max_seq_len=16)

    # Cria Q e K em bf16
    q = torch.randn(1, 2, 8, 32).to(torch.bfloat16)
    k = torch.randn(1, 2, 8, 32).to(torch.bfloat16)

    q_rot, k_rot = rope(q, k)

    # Deve manter dtype bf16
    print(f"   Dtype Q rotacionado: {q_rot.dtype}")
    assert q_rot.dtype == torch.bfloat16

    # Deve preservar norma (rotação é isometria)
    norm_antes = q.float().norm(dim=-1).mean().item()
    norm_depois = q_rot.float().norm(dim=-1).mean().item()
    print(f"   Norma antes:  {norm_antes:.4f}")
    print(f"   Norma depois: {norm_depois:.4f}")
    assert abs(norm_antes - norm_depois) < 0.05, "RoPE quebrou em bf16!"
    print("   ✓ OK — RoPE é bf16-safe (Regra 7 respeitada)")


def test_rope_relative_positions():
    print("\n── Teste 8: RoPE — posições relativas ─────────────────")
    rope = RoPE(head_dim=32, max_seq_len=64)

    q_a = torch.randn(1, 1, 1, 32)
    k_a = torch.randn(1, 1, 1, 32)

    seq_a_q = torch.randn(1, 1, 6, 32)
    seq_a_k = torch.randn(1, 1, 6, 32)
    seq_a_q[:, :, 0] = q_a.squeeze(2)
    seq_a_k[:, :, 5] = k_a.squeeze(2)

    q_rot, k_rot = rope(seq_a_q, seq_a_k)
    dot_a = (q_rot[:, :, 0] * k_rot[:, :, 5]).sum().item()

    seq_b_q = torch.randn(1, 1, 12, 32)
    seq_b_k = torch.randn(1, 1, 12, 32)
    seq_b_q[:, :, 3] = q_a.squeeze(2)
    seq_b_k[:, :, 8] = k_a.squeeze(2)

    q_rot_b, k_rot_b = rope(seq_b_q, seq_b_k)
    dot_b = (q_rot_b[:, :, 3] * k_rot_b[:, :, 8]).sum().item()

    print(f"   Produto interno (pos 0 e 5):  {dot_a:.6f}")
    print(f"   Produto interno (pos 3 e 8):  {dot_b:.6f}")
    assert abs(dot_a - dot_b) < 1e-4
    print("   ✓ OK — invariância relativa confirmada")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTE DO RMSNORM + ROPE v2 — Luna")
    print("=" * 60)

    test_rmsnorm_shape()
    test_rmsnorm_normalizes()
    test_rmsnorm_backward()
    test_rope_freqs()
    test_rope_shape()
    test_rope_preserves_norm()
    test_rope_bf16_safe()
    test_rope_relative_positions()

    print("\n" + "=" * 60)
    print("✅ TODOS OS TESTES PASSARAM")
    print("=" * 60)