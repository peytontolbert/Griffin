import torch

from griffin import GatedMLPBlock


def test_gated_mlp_block_preserves_input_width():
    block = GatedMLPBlock(input_dim=10, hidden_dim=30)
    x = torch.randn(4, 6, 10)

    output = block(x)

    assert output.shape == x.shape
    assert block.linear1.weight.shape == (30, 10)
    assert block.linear2.weight.shape == (30, 10)
    assert block.linear3.weight.shape == (10, 30)
