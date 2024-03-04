import torch
from torch import nn
from griffin import GriffinModel

# Test GriffinModel
model = GriffinModel(input_dim=1024, mlp_expansion_factor=3, rnn_width=1536, depth=12)
x = torch.randn(1, 32, 1024)
output = model.forward(x)
assert output.shape == (32, 1024)
