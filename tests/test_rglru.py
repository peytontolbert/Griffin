import torch
from griffin import RG_LRU
rnn_width=100
batch_size = 32
# Create an instance of RG_LRU
module = RG_LRU(rnn_width)

# Test forward pass
xt = torch.randn(batch_size, rnn_width)  # Input tensor of shape (batch_size, input_dim)
ht_minus_1 = torch.randn(
    batch_size, rnn_width
)  # Previous hidden state tensor of shape (batch_size, hidden_dim)
output = module.forward(xt, ht_minus_1)
print(output.shape)  # Expected output: (32, 20)
