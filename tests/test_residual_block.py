"""Tests for residual-block shape and component wiring."""

import torch

from griffin import GatedMLPBlock, RMSNorm, RecurrentBlock, ResidualBlock


def test_residual_block_preserves_input_shape_and_components():
    """A residual block should keep shape and expose expected submodules."""
    block = ResidualBlock(input_dim=10, expansion_factor=3, rnn_width=13)
    x = torch.randn(2, 7, 10)

    output = block(x)

    assert output.shape == x.shape
    assert isinstance(block.mlp, GatedMLPBlock)
    assert isinstance(block.norm1, RMSNorm)
    assert isinstance(block.recurrent, RecurrentBlock)
    assert isinstance(block.norm2, RMSNorm)
