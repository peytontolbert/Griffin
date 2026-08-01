"""Behavioral tests for causal local multi-query attention."""

import torch

from griffin import LocalMQAAttention


def test_local_attention_is_causal():
    """Changing a future token must not affect earlier attention outputs."""
    torch.manual_seed(0)
    attention = LocalMQAAttention(input_dim=8, num_heads=2, window_size=4).eval()
    x = torch.randn(2, 7, 8)
    changed = x.clone()
    changed[:, 5] += 10.0

    with torch.no_grad():
        original = attention(x)
        modified = attention(changed)

    torch.testing.assert_close(original[:, :5], modified[:, :5])


def test_local_attention_cache_matches_full_sequence():
    """Bounded KV caching should match a full sliding-window forward pass."""
    torch.manual_seed(0)
    attention = LocalMQAAttention(input_dim=8, num_heads=2, window_size=4).eval()
    x = torch.randn(2, 9, 8)

    with torch.no_grad():
        full = attention(x)
        first, cache = attention(x[:, :2], return_cache=True)
        second, cache = attention(x[:, 2:6], cache, return_cache=True)
        third, cache = attention(x[:, 6:], cache, return_cache=True)

    assert cache.keys.shape[1] == 3
    assert cache.values.shape[1] == 3
    torch.testing.assert_close(
        full,
        torch.cat([first, second, third], dim=1),
        atol=1e-6,
        rtol=1e-6,
    )
