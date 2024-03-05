import torch
from torch import nn
from griffin.griffin import GatedMLPBlock

input_dim = 10
hidden_dim = 10 * 3
batch_size = 45
# Create an instance of GatedMLPBlock
block = GatedMLPBlock(input_dim, hidden_dim)

# Test forward pass
x = torch.randn(batch_size, input_dim)  # Input tensor of shape (batch_size, input_dim)
output = block.forward(x)
print(output.shape)  # Expected output: (32, 10)

# Test the shape of linear layers
print(block.linear1.weight.shape)  # Expected output: (30, 10)
print(block.linear2.weight.shape)  # Expected output: (30, 10)
print(block.linear3.weight.shape)  # Expected output: (10, 30)
