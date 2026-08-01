"""Small evaluation-mode smoke test for the Griffin-style language model."""

import torch

from griffin import GriffinModel


# Evaluation should be deterministic for identical token IDs.
vocab_size = 100
model = GriffinModel(vocab_size=vocab_size, input_dim=64, rnn_width=64, depth=2)
model.eval()

token_ids = torch.randint(0, vocab_size, (2, 4))
with torch.no_grad():
    logits = model(token_ids)

print("Logits shape:", logits.shape)
