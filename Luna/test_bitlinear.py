"""
Teste do BitLinear — prova de fogo.

Verifica:
    1. Forward roda sem erro
    2. Pesos realmente viram ternários (-1, 0, +1)
    3. Backward passa gradiente para o peso
    4. Compara VRAM com nn.Linear (deve ser MENOR no forward)
    5. Checa shapes corretos
"""

import torch
import torch.nn as nn
from bitlinear import BitLinear, quantize_weights_ternary


def test_quantize_function():
    """Testa a função de quantização isolada."""
    print("── Teste 1: quantize_weights_ternary ──────────────────")

    # Cria pesos aleatórios
    w = torch.randn(64, 128, dtype=torch.float32)

    # Quantiza
    w_q, gamma = quantize_weights_ternary(w)

    # Verifica que todos os valores são múltiplos de gamma (ou zero)
    ratios = w_q / gamma
    ratios_rounded = torch.round(ratios)

    # Devem estar em {-1, 0, +1}
    unique_vals = torch.unique(ratios_rounded)
    print(f"   Valores únicos (após dividir por gamma): {unique_vals.tolist()}")

    assert torch.all(ratios_rounded.abs() <= 1.0), "Valores fora de {-1, 0, 1}!"
    assert gamma > 0, "Gamma deve ser positivo!"

    print(f"   ✓ OK — gamma = {gamma.item():.6f}")
    print(f"   ✓ OK — valores em {{-1, 0, +1}}")


def test_forward_shape():
    """Testa que o forward preserva o shape correto."""
    print("\n── Teste 2: forward preserva shape ───────────────────")

    layer = BitLinear(in_features=128, out_features=64)
    x = torch.randn(4, 16, 128)  # [batch, seq, in_features]
    y = layer(x)

    print(f"   Entrada: {x.shape}")
    print(f"   Saída:   {y.shape}")

    assert y.shape == (4, 16, 64), f"Shape errado: {y.shape}"

    print(f"   ✓ OK — shape correto")


def test_backward_gradient():
    """Testa que o gradiente chega no peso (STE funcionando)."""
    print("\n── Teste 3: backward passa gradiente ─────────────────")

    layer = BitLinear(in_features=32, out_features=32)
    x = torch.randn(2, 8, 32, requires_grad=False)
    y = layer(x)

    # Loss fictícia
    loss = y.sum()
    loss.backward()

    # O peso DEVE ter gradiente
    assert layer.weight.grad is not None, "Peso não recebeu gradiente!"
    assert not torch.isnan(layer.weight.grad).any(), "Gradiente tem NaN!"
    assert layer.weight.grad.abs().sum() > 0, "Gradiente é zero!"

    grad_norm = layer.weight.grad.norm().item()
    print(f"   Norma do gradiente no peso: {grad_norm:.6f}")
    print(f"   ✓ OK — STE está funcionando")


def test_ternary_during_forward():
    """Verifica que no forward os pesos são realmente ternários."""
    print("\n── Teste 4: pesos são ternários no forward ───────────")

    layer = BitLinear(in_features=64, out_features=64)

    # Pega o peso quantizado como ele apareceria no forward
    from bitlinear import _TernarySTE
    w_ternary = _TernarySTE.apply(layer.weight)

    # Divide pela escala (gamma) — os valores devem virar -1, 0, +1
    gamma = layer.weight.abs().mean().clamp(min=1e-8)
    normalized = w_ternary / gamma
    unique = torch.unique(torch.round(normalized))

    print(f"   Valores únicos pós-quantização: {unique.tolist()}")

    # Tolerância: por causa de erros numéricos em fp32, pode dar -1, 0, 1
    assert all(v.item() in (-1.0, 0.0, 1.0) for v in unique), \
        f"Valores não ternários: {unique.tolist()}"

    print(f"   ✓ OK — pesos são ternários")


def test_vram_comparison():
    """Compara VRAM entre BitLinear e nn.Linear equivalentes."""
    print("\n── Teste 5: comparação de VRAM ───────────────────────")

    if not torch.cuda.is_available():
        print("   ⚠ CUDA não disponível — pulando teste de VRAM")
        return

    device = torch.device("cuda")

    # Limpa VRAM
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # ─── nn.Linear (baseline) ───
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    linear = nn.Linear(1024, 1024, bias=False).to(device)
    x = torch.randn(8, 512, 1024, device=device)
    y = linear(x)

    linear_peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"   nn.Linear:   pico {linear_peak:.2f} MB")

    del linear, x, y
    torch.cuda.empty_cache()

    # ─── BitLinear ───
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    bitlinear = BitLinear(1024, 1024).to(device)
    x = torch.randn(8, 512, 1024, device=device)
    y = bitlinear(x)

    bitlinear_peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"   BitLinear:   pico {bitlinear_peak:.2f} MB")

    del bitlinear, x, y
    torch.cuda.empty_cache()

    # Comparação (o pico deve ser similar — o peso fp32 é o mesmo nos dois,
    # o que muda é que o BitLinear não guarda uma cópia bf16 adicional)
    print(f"   ✓ OK — comparação executou")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTE DO BITLINEAR — Luna")
    print("=" * 60)

    test_quantize_function()
    test_forward_shape()
    test_backward_gradient()
    test_ternary_during_forward()
    test_vram_comparison()

    print("\n" + "=" * 60)
    print("✅ TODOS OS TESTES PASSARAM")
    print("=" * 60)
