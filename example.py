import torch

from griffin import GriffinModel


vocab_size = 100
model = GriffinModel(
    vocab_size=vocab_size,
    input_dim=64,
    mlp_expansion_factor=2,
    rnn_width=64,
    depth=2,
)
token_ids = torch.randint(0, vocab_size, (3, 15))
logits = model(token_ids)

print("Model output shape:", logits.shape)
