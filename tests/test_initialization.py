"""Tests for the reference-style Griffin parameter initialization."""

import math

import torch

from griffin import GriffinModel


def _assert_std_near(weight: torch.Tensor, expected: float, relative_tolerance: float = 0.15) -> None:
    """Assert that an initialized tensor's sample deviation is near its target."""
    actual = weight.std().item()
    assert abs(actual - expected) <= expected * relative_tolerance


def test_model_initializes_all_projection_families_explicitly():
    """Recurrent, convolution, attention, and MLP weights should match reference scales."""
    torch.manual_seed(0)
    input_dim = 128
    rnn_width = 192
    depth = 3
    model = GriffinModel(
        vocab_size=257,
        input_dim=input_dim,
        mlp_expansion_factor=3,
        rnn_width=rnn_width,
        depth=depth,
        attention_heads=4,
    )
    recurrent = model.layers[0].recurrent
    attention = model.layers[2].attention
    mlp = model.layers[0].mlp
    residual_scale = 2.0 / depth

    _assert_std_near(recurrent.linear_x.weight, math.sqrt(1.0 / input_dim))
    _assert_std_near(recurrent.linear_y.weight, math.sqrt(1.0 / input_dim))
    _assert_std_near(recurrent.temporal_conv.weight, math.sqrt(0.01 / 4))
    _assert_std_near(recurrent.linear_out.weight, math.sqrt(residual_scale / rnn_width))
    _assert_std_near(attention.query.weight, math.sqrt(1.0 / input_dim))
    _assert_std_near(attention.key.weight, math.sqrt(1.0 / input_dim))
    _assert_std_near(attention.value.weight, math.sqrt(1.0 / input_dim))
    _assert_std_near(attention.output.weight, math.sqrt(residual_scale / input_dim))
    _assert_std_near(mlp.linear1.weight, math.sqrt(1.0 / input_dim))
    _assert_std_near(mlp.linear2.weight, math.sqrt(1.0 / input_dim))
    _assert_std_near(mlp.linear3.weight, math.sqrt(residual_scale / mlp.linear3.in_features))

    for bias in (
        recurrent.linear_x.bias,
        recurrent.linear_y.bias,
        recurrent.temporal_conv.bias,
        recurrent.linear_out.bias,
        attention.output.bias,
        mlp.linear1.bias,
        mlp.linear2.bias,
        mlp.linear3.bias,
    ):
        torch.testing.assert_close(bias, torch.zeros_like(bias))
