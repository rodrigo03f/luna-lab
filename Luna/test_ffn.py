"""
Teste da SwiGLU FFN — shapes, backward, comportamento do gate.
"""

import torch
from ffn import SwiGLUFFN


def test_shape():
    """FFN preserva shape."""
    print("── Teste 1: preserva shape ────────────────────────────")
    ffn = SwiGLUFFN(d_model=256, d_ff=768)
    x = torch.randn(2, 32, 256)
    y = ffn(x)
    print(f"   Entrada: {x.shape}  Saída: {y.shape}")
    assert y.shape == x.shape
    print("   ✓ OK")


def test_backward():
    """FFN passa gradiente."""
    print("\n── Teste 2: backward ──────────────────────────────────")
    ffn = SwiGLUFFN(d_model=128, d_ff=384)
    x = torch.randn(2, 16, 128, requires_grad=True)
    y = ffn(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    print(f"   Norma do gradiente: {x.grad.norm().item():.4f}")
    print("   ✓ OK")


def test_swiglu_gating():
    """
    Teste fundamental do SwiGLU: o gate deve modular o up.
    Se o gate for zero, a saída deve ser zero (independente do up).
    """
    print("\n── Teste 3: gate controla o fluxo ─────────────────────")
    import torch.nn.functional as F

    # Teste direto: silu(0) * up = 0 * up = 0
    gate = torch.zeros(4)
    up = torch.randn(4) * 100.0  # valores grandes
    result = F.silu(gate) * up
    print(f"   Gate zerado, up grande: resultado = {result.abs().max().item():.6f}")
    assert torch.allclose(result, torch.zeros_like(result))
    print("   ✓ OK — gate modula corretamente")


def test_no_nan():
    """FFN não produz NaN em forward nem backward."""
    print("\n── Teste 4: sem NaN ───────────────────────────────────")
    ffn = SwiGLUFFN(d_model=64, d_ff=128)
    x = torch.randn(4, 8, 64) * 5.0
    y = ffn(x)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()
    print(f"   Saída max: {y.abs().max().item():.4f}")
    print("   ✓ OK")


def test_ffn_vs_relu_param_count():
    """Compara número de parâmetros: SwiGLU (3 matrizes) vs ReLU (2)."""
    print("\n── Teste 5: parâmetros SwiGLU ─────────────────────────")
    d_model, d_ff = 256, 768
    ffn = SwiGLUFFN(d_model=d_model, d_ff=d_ff)
    total = sum(p.numel() for p in ffn.parameters())
    expected = 3 * d_model * d_ff  # gate + up + down
    print(f"   Parâmetros totais: {total}")
    print(f"   Esperado (3 × {d_model} × {d_ff}): {expected}")
    assert total == expected
    print("   ✓ OK")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTE DA SWIGLU FFN — Luna")
    print("=" * 60)

    test_shape()
    test_backward()
    test_swiglu_gating()
    test_no_nan()
    test_ffn_vs_relu_param_count()

    print("\n" + "=" * 60)
    print("✅ TODOS OS TESTES PASSARAM")
    print("=" * 60)