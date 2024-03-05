import torch
from griffin.griffin import RMSNorm

# Test case 1: Test forward pass with a random input tensor
input_dim = 10
x = torch.randn(32, input_dim)  # Input tensor of shape (batch_size, input_dim)
norm_layer = RMSNorm(input_dim)
output = norm_layer.forward(x)
print(output.shape)  # Expected output: (32, 10)

# Test case 2: Test the scale parameter
print(norm_layer.scale)  # Expected output: 3.1622776601683795

# Test case 3: Test the g parameter
print(
    norm_layer.g
)  # Expected output: tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], requires_grad=True)
