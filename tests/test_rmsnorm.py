"""Tests for RMS normalization behavior."""

import torch

from griffin import RMSNorm


def test_rmsnorm_preserves_shape_and_normalizes_rms():
    """RMSNorm should preserve shape and produce unit RMS with default scale."""
    norm = RMSNorm(10)
    x = torch.randn(4, 6, 10)

    output = norm(x)

    rms = output.pow(2).mean(dim=-1).sqrt()
    assert output.shape == x.shape
    torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-5, rtol=1e-5)
