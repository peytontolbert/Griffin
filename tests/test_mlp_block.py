"""Tests for the gated MLP residual sublayer."""

import torch
import torch.nn.functional as F

from griffin import GatedMLPBlock


def test_gated_mlp_block_preserves_input_width():
    """The MLP expands internally and projects back to the input width."""
    block = GatedMLPBlock(input_dim=10, hidden_dim=30)
    x = torch.randn(4, 6, 10)

    output = block(x)

    assert output.shape == x.shape
    assert block.linear1.weight.shape == (30, 10)
    assert block.linear2.weight.shape == (30, 10)
    assert block.linear3.weight.shape == (10, 30)


def test_gated_mlp_uses_reference_tanh_gelu():
    """The MLP gate should use the tanh GELU approximation used by JAX."""
    torch.manual_seed(0)
    block = GatedMLPBlock(input_dim=6, hidden_dim=12)
    x = torch.randn(2, 4, 6)

    output = block(x)
    expected_gate = F.gelu(block.linear1(x), approximate="tanh")
    expected = block.linear3(expected_gate * block.linear2(x))

    torch.testing.assert_close(output, expected)
