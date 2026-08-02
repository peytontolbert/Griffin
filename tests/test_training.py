"""Tests for optimizer parameter grouping used by the training program."""

import torch

from griffin import GriffinModel, RMSNorm
from train import adamw_parameter_groups


def test_adamw_excludes_biases_and_norm_offsets_from_weight_decay():
    """AdamW should decay matrix weights but not biases or RMSNorm offsets."""
    model = GriffinModel(
        vocab_size=17,
        input_dim=8,
        mlp_expansion_factor=2,
        rnn_width=12,
        depth=3,
        attention_heads=2,
    )
    groups = adamw_parameter_groups(model, weight_decay=0.1)
    decayed = {id(parameter) for parameter in groups[0]["params"]}
    not_decayed = {id(parameter) for parameter in groups[1]["params"]}
    all_trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}

    assert groups[0]["weight_decay"] == 0.1
    assert groups[1]["weight_decay"] == 0.0
    assert decayed.isdisjoint(not_decayed)
    assert decayed | not_decayed == all_trainable
    assert id(model.embd.weight) in decayed

    for module in model.modules():
        if isinstance(module, RMSNorm):
            assert id(module.g) in not_decayed
        bias = getattr(module, "bias", None)
        if isinstance(bias, torch.nn.Parameter):
            assert id(bias) in not_decayed
