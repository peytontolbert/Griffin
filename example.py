"""Minimal forward-pass example for the Griffin-style language model."""

import torch

from griffin import GriffinModel


# The model expects integer token IDs and returns raw vocabulary logits.
vocab_size = 100
model = GriffinModel(
    vocab_size=vocab_size,
    input_dim=64,
    mlp_expansion_factor=2,
    rnn_width=64,
    # Three layers instantiate one complete recurrent/recurrent/attention group.
    depth=3,
)
token_ids = torch.randint(0, vocab_size, (3, 15))
logits = model(token_ids)

print("Model output shape:", logits.shape)
