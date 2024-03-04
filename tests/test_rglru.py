import torch
from griffin import RG_LRU

# Create an instance of RG_LRU
module = RG_LRU(input_dim=10, mult=3)

# Test forward pass
xt = torch.randn(32, 10)  # Input tensor of shape (batch_size, input_dim)
ht_minus_1 = torch.randn(
    32, 10
)  # Previous hidden state tensor of shape (batch_size, hidden_dim)
output = module.forward(xt, ht_minus_1)
print(output.shape)  # Expected output: (32, 20)
