"""
Teste da Attention — máscara causal, shapes, backward.
"""

import torch
from attention import Attention


def test_shape():
    """Attention preserva shape."""
    print("── Teste 1: preserva shape ────────────────────────────")
    attn = Attention(d_model=256, n_heads=4, max_seq_len=512)
    x = torch.randn(2, 32, 256)
    y = attn(x)
    print(f"   Entrada: {x.shape}  Saída: {y.shape}")
    assert y.shape == x.shape
    print("   ✓ OK")


def test_backward():
    """Attention passa gradiente."""
    print("\n── Teste 2: backward ──────────────────────────────────")
    attn = Attention(d_model=128, n_heads=4, max_seq_len=64)
    x = torch.randn(2, 16, 128, requires_grad=True)
    y = attn(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    print(f"   Norma do gradiente: {x.grad.norm().item():.4f}")
    print("   ✓ OK")


def test_causal_deterministic():
    """Mesma entrada → mesma saída (sem dropout, sem randomness)."""
    print("\n── Teste 3: determinístico ────────────────────────────")
    attn = Attention(d_model=64, n_heads=4, max_seq_len=32)
    attn.eval()
    x = torch.randn(1, 8, 64)
    y1 = attn(x)
    y2 = attn(x)
    diff = (y1 - y2).abs().max().item()
    print(f"   Diferença entre duas execuções: {diff:.8f}")
    assert diff < 1e-6
    print("   ✓ OK")


def test_causal_mask_blocks_future():
    """
    TESTE CRÍTICO: a máscara causal deve impedir que o token na posição i
    veja o token na posição j > i.

    Verificamos assim: mudamos o valor de um token FUTURO (posição 1) e
    checamos que a saída do token PASSADO (posição 0) NÃO muda.
    """
    print("\n── Teste 4: máscara causal bloqueia futuro ────────────")
    attn = Attention(d_model=64, n_heads=4, max_seq_len=32)
    attn.eval()

    x = torch.randn(1, 4, 64)

    # Saída com x original
    y_original = attn(x)

    # Modifica APENAS o token na posição 1 (futuro do token 0)
    x_modified = x.clone()
    x_modified[0, 1, :] = torch.randn(64) * 10.0  # valor bem diferente

    y_modified = attn(x_modified)

    # Saída do token 0 NÃO deve mudar (ele não pode ver o token 1)
    diff_pos0 = (y_original[0, 0] - y_modified[0, 0]).abs().max().item()

    # Saída do token 1 DEVE mudar (ele pode ver a si mesmo e o token 0,
    # e a própria entrada dele mudou)
    diff_pos1 = (y_original[0, 1] - y_modified[0, 1]).abs().max().item()

    print(f"   Diff no token 0 (não deve mudar): {diff_pos0:.8f}")
    print(f"   Diff no token 1 (deve mudar):     {diff_pos1:.6f}")

    assert diff_pos0 < 1e-6, "Máscara causal FALHOU: token 0 foi afetado pelo token 1!"
    assert diff_pos1 > 1e-4, "Token 1 não mudou — algo está errado."

    print("   ✓ OK — máscara causal funciona")


def test_attention_weights_sum_to_one():
    """Após softmax, os pesos de atenção devem somar 1."""
    print("\n── Teste 5: pesos somam 1 ─────────────────────────────")
    attn = Attention(d_model=64, n_heads=4, max_seq_len=16)
    attn.eval()

    x = torch.randn(1, 8, 64)

    # Replica o forward até o softmax pra inspecionar os pesos
    batch, seq_len, _ = x.shape
    qkv = attn.qkv_proj(x)
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(batch, seq_len, attn.n_heads, attn.head_dim).transpose(1, 2)
    k = k.view(batch, seq_len, attn.n_heads, attn.head_dim).transpose(1, 2)
    q, k = attn.rope(q, k)
    scores = torch.matmul(q, k.transpose(-2, -1)) * attn.scale
    scores = scores + attn._get_causal_mask(seq_len, x.device)
    weights = torch.softmax(scores.float(), dim=-1)

    # Verifica que somam 1 (exceto onde -inf → softmax dá 0)
    sums = weights.sum(dim=-1)
    print(f"   Soma mínima: {sums.min().item():.6f}")
    print(f"   Soma máxima: {sums.max().item():.6f}")
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    print("   ✓ OK — pesos somam 1")


def test_head_split():
    """Testa que as cabeças veem partes diferentes da representação."""
    print("\n── Teste 6: cabeças independentes ─────────────────────")
    attn = Attention(d_model=64, n_heads=4, max_seq_len=8)
    attn.eval()

    # Uma entrada constante — sem dropout, as saídas de cada cabeça
    # devem ser diferentes (porque os pesos das cabeças são diferentes).
    x = torch.ones(1, 4, 64)

    qkv = attn.qkv_proj(x)
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(1, 4, attn.n_heads, attn.head_dim).transpose(1, 2)

    # Verifica que as cabeças NÃO são idênticas
    head0 = q[0, 0].flatten()
    head1 = q[0, 1].flatten()
    diff = (head0 - head1).abs().mean().item()
    print(f"   Diferença média entre cabeça 0 e 1: {diff:.6f}")
    assert diff > 1e-4
    print("   ✓ OK — cabeças processam informação diferente")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTE DA ATTENTION — Luna")
    print("=" * 60)

    test_shape()
    test_backward()
    test_causal_deterministic()
    test_causal_mask_blocks_future()
    test_attention_weights_sum_to_one()
    test_head_split()

    print("\n" + "=" * 60)
    print("✅ TODOS OS TESTES PASSARAM")
    print("=" * 60)