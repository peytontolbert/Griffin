import torch
from griffin import RecurrentBlock

# Test case 1: Test forward pass
input_dim = 10
hidden_dim = 30
rnn_width = 10*4/3
block = RecurrentBlock(input_dim, hidden_dim, rnn_width)
x = torch.randn(32, hidden_dim)  # Input tensor of shape (batch_size, input_dim)
output = block.forward(x)
print(output.shape)  # Expected output: (32, 10)

# Test case 2: Test the shape of linear layers
print(block.linear.weight.shape)  # Expected output: (10, 20)

# Test case 3: Test the shape of temporal_conv layer
print(block.temporal_conv.weight.shape)  # Expected output: (20, 10, 1)

# Test case 4: Test the shape of rg_lru layer
print(block.rg_lru.linear1.weight.shape)  # Expected output: (20, 20)
print(block.rg_lru.linear2.weight.shape)  # Expected output: (20, 20)
print(block.rg_lru.linear3.weight.shape)  # Expected output: (20, 20)
