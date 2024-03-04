from griffin import GriffinModel
import torch

# Example usage
input_dim = 512  # example dimension
hidden_dim = 512  # example dimension
num_blocks = 3  # example number of blocks
model = GriffinModel(input_dim, hidden_dim, num_blocks)
inputs = torch.randn(3, 15, input_dim)  # example input tensor
output = model(inputs)

print("Model Output Shape:", output.shape)
