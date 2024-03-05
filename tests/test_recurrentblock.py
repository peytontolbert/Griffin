import torch
from torch import nn
from griffin.griffin import GriffinModel, RecurrentBlock


def test_recurrent_block():
    # Create an instance of the RecurrentBlock class
    block = RecurrentBlock(input_dim=10, rnn_width=13)

    # Create a random input tensor
    x = torch.randn(1, 32, 10)

    # Call the forward method of the RecurrentBlock class
    output = block.forward(x)

    # Check if the output tensor has the expected shape
    return output


# Run the test
output = test_recurrent_block()
print(f"output: {output.shape}")
