"""
Teste do LOTTriton — kernel fundido.

Valida:
    1. Kernel roda sem erro em CUDA
    2. Resultado bate com LOT v1.2 (PyTorch puro)
    3. Sem NaN
    4. Velocidade é maior que v1.2 (esse é o ponto)
"""

import time

import torch

from optim.lot import LOT
from optim.lot_triton import LOTTriton


def test_triton_runs():
    """Kernel Triton roda sem erro."""
    print("── Teste 1: kernel Triton roda ────────────────────────")

    if not torch.cuda.is_available():
        print("   ⚠ CUDA não disponível — pulando")
        return

    p = torch.nn.Parameter(torch.randn(4096, device="cuda"))
    opt = LOTTriton([p], lr=0.01)

    p.grad = torch.randn_like(p) * 0.1
    opt.step()

    assert not torch.isnan(p).any()
    print(f"   Pesos após 1 step: min={p.min().item():.4f}, max={p.max().item():.4f}")
    print("   ✓ OK")


def test_matches_v1_2():
    """LOTTriton bate com LOT v1.2 (matemática idêntica)."""
    print("\n── Teste 2: LOTTriton bate com LOT v1.2 ───────────────")

    if not torch.cuda.is_available():
        print("   ⚠ CUDA não disponível — pulando")
        return

    torch.manual_seed(42)
    p_v1 = torch.nn.Parameter(torch.randn(10_000, device="cuda"))
    p_v2 = torch.nn.Parameter(p_v1.clone().detach())

    opt_v1 = LOT([p_v1], lr=0.01, betas=(0.9, 0.99), weight_decay=0.0)
    opt_v2 = LOTTriton([p_v2], lr=0.01, betas=(0.9, 0.99), weight_decay=0.0)

    max_diff = 0.0
    for step in range(10):
        torch.manual_seed(100 + step)
        g = torch.randn(10_000, device="cuda") * 0.1

        p_v1.grad = g.clone()
        p_v2.grad = g.clone()

        opt_v1.step()
        opt_v2.step()

        diff = (p_v1 - p_v2).abs().max().item()
        max_diff = max(max_diff, diff)
        print(f"   Step {step + 1:2d}: diff máxima = {diff:.8f}")

    print(f"\n   Diferença máxima: {max_diff:.8f}")
    # Deve ser 0 (ou quase — erro de floating point)
    assert max_diff < 1e-4, f"Triton e v1.2 divergem: {max_diff}"
    print("   ✓ OK — kernel implementa a mesma matemática")


def test_no_nan_100_steps():
    """Kernel roda 100 steps sem NaN/Inf."""
    print("\n── Teste 3: 100 steps sem NaN ─────────────────────────")

    if not torch.cuda.is_available():
        print("   ⚠ CUDA não disponível — pulando")
        return

    p = torch.nn.Parameter(torch.randn(4096, device="cuda"))
    opt = LOTTriton([p], lr=1e-3)

    for _ in range(100):
        p.grad = torch.randn(4096, device="cuda") * 10.0
        opt.step()

    assert not torch.isnan(p).any()
    assert not torch.isinf(p).any()
    print(f"   Pesos finais: min={p.min().item():.4f}, max={p.max().item():.4f}")
    print("   ✓ OK")


def test_speedup():
    """Velocidade do LOTTriton > LOT v1.2."""
    print("\n── Teste 4: velocidade ────────────────────────────────")

    if not torch.cuda.is_available():
        print("   ⚠ CUDA não disponível — pulando")
        return

    N = 1_000_000  # tensor grande pra medir
    N_STEPS = 50

    # Setup v1.2
    torch.manual_seed(42)
    p_v1 = torch.nn.Parameter(torch.randn(N, device="cuda"))
    opt_v1 = LOT([p_v1], lr=0.01)

    # Setup v2 (Triton)
    torch.manual_seed(42)
    p_v2 = torch.nn.Parameter(p_v1.clone().detach())
    opt_v2 = LOTTriton([p_v2], lr=0.01)

    # Warmup (1 step cada)
    p_v1.grad = torch.randn(N, device="cuda") * 0.1
    opt_v1.step()
    p_v2.grad = torch.randn(N, device="cuda") * 0.1
    opt_v2.step()

    # Mede v1.2
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N_STEPS):
        p_v1.grad = torch.randn(N, device="cuda") * 0.1
        opt_v1.step()
    torch.cuda.synchronize()
    t_v1 = time.time() - t0

    # Mede v2
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N_STEPS):
        p_v2.grad = torch.randn(N, device="cuda") * 0.1
        opt_v2.step()
    torch.cuda.synchronize()
    t_v2 = time.time() - t0

    print(f"   LOT v1.2 (PyTorch):  {t_v1 * 1000:.1f} ms ({t_v1 / N_STEPS * 1000:.3f} ms/step)")
    print(f"   LOTTriton (kernel):  {t_v2 * 1000:.1f} ms ({t_v2 / N_STEPS * 1000:.3f} ms/step)")
    print(f"   Speedup:             {t_v1 / t_v2:.2f}x")

    assert t_v2 < t_v1, "Triton NÃO foi mais rápido!"
    print("   ✓ OK")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTE DO LOTTRITON")
    print("=" * 60)

    test_triton_runs()
    test_matches_v1_2()
    test_no_nan_100_steps()
    test_speedup()

    print("\n" + "=" * 60)
    print("✅ TODOS OS TESTES PASSARAM")
    print("=" * 60)