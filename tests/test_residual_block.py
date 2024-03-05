from griffin import ResidualBlock, GatedMLPBlock, RMSNorm, RecurrentBlock
import torch
def test_residual_block():
    # Create an instance of the ResidualBlock class
    block = ResidualBlock(input_dim=10, expansion_factor=3, rnn_width=13)

    # Create a random input tensor
    x = torch.randn(1, 32, 10)

    # Call the forward method of the ResidualBlock class
    output = block.forward(x)

    # Check if the output tensor has the expected shape
    assert output.shape == (1, 32, 10)

    # Check if the residual shape is the same as the input shape
    assert output.shape == x.shape

    # Check if the mlp attribute is an instance of GatedMLPBlock
    assert isinstance(block.mlp, GatedMLPBlock)

    # Check if the norm1 attribute is an instance of RMSNorm
    assert isinstance(block.norm1, RMSNorm)

    # Check if the recurrent attribute is an instance of RecurrentBlock
    assert isinstance(block.recurrent, RecurrentBlock)

    # Check if the norm2 attribute is an instance of RMSNorm
    assert isinstance(block.norm2, RMSNorm)

    print("All tests passed.")
    return output

# Run the test
output = test_residual_block()
print(f"Output shape: {output.shape}")