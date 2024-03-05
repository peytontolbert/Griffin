import torch
from torch.nn import Embedding
import torch.nn as nn
from griffin import GriffinModel, RecurrentBlock

vocab_size = 100


def test_griffin_model():
    # Create an instance of the GriffinModel class
    model = GriffinModel(
        vocab_size=100, input_dim=10, mlp_expansion_factor=3, rnn_width=13, depth=12
    )

    # Create a random input tensor
    x = torch.randint(1, 100, (1, 32))

    # Call the forward method of the GriffinModel class
    output = model.forward(x)

    # Check if the output tensor has the expected shape
    assert output.shape == (1, 32, 10)

    # Check if the layers attribute is an instance of nn.ModuleList
    assert isinstance(model.layers, nn.ModuleList)

    # Check if the embd attribute is an instance of Embedding
    assert isinstance(model.embd, Embedding)

    print("All tests passed.")
    return output


# Run the test
output = test_griffin_model()
print(f"Output shape: {output.shape}")
