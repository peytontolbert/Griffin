import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Embedding


def output_head(x: Tensor, dim: int):
    """
    Applies a linear transformation followed by softmax activation to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim).
        dim (int): Dimension of the input tensor.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, dim) after applying linear transformation and softmax activation.
    """
    x = nn.LayerNorm(dim)(x)  # Adds training stability

    x = nn.Linear(dim, dim)(x)  # Linear transformation

    return F.softmax(x, dim=-1)  # Softmax


class MLPBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPBlock, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        """
        Initializes an instance of RMSNorm.

        Args:
            dim (int): The dimension of the input tensor.

        """
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """
        Performs forward pass of the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized and scaled tensor.

        """
        return F.normalize(x, dim=-1) * self.scale * self.g


class ResidualBlock(nn.Module):
    """
    Residual block module that performs residual connections and applies normalization.

    Args:
        input_dim (int): The input dimension of the block.
        hidden_dim (int): The hidden dimension of the block.

    Returns:
        torch.Tensor: The output tensor after applying the residual block.
    """

    def __init__(self, input_dim, hidden_dim, rnn_width):
        super(ResidualBlock, self).__init__()
        self.mlp = GatedMLPBlock(input_dim, hidden_dim)
        self.norm1 = RMSNorm(hidden_dim)
        self.recurrent = RecurrentBlock(input_dim, hidden_dim, rnn_width)
        self.norm2 = RMSNorm(hidden_dim)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.recurrent(x)
        residual = x + residual
        x = self.norm2(residual)
        x = self.mlp(x)
        return x + residual


class GatedMLPBlock(nn.Module):
    """
    Gated MLP Block class.

    This class represents a gated MLP block, which consists of three linear layers.
    The input dimension is the size of the input tensor, and the hidden dimension is the size of the hidden layer.

    Args:
        input_dim (int): The size of the input tensor.
        hidden_dim (int): The size of the hidden layer.

    Attributes:
        linear1 (nn.Linear): The first linear layer.
        linear2 (nn.Linear): The second linear layer.
        linear3 (nn.Linear): The third linear layer.
    """

    def __init__(self, input_dim, hidden_dim):
        super(GatedMLPBlock, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(input_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        """
        Forward pass of the GatedMLPBlock.

        Applies the gated MLP block to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the gated MLP block.
        """
        output = torch.sigmoid(self.linear1(x))
        gate = F.gelu(output)
        combined = gate * self.linear2(x)
        return self.linear3(combined)


class RG_LRU(nn.Module):
    def __init__(self, input_dim, mult):
        """
        Initializes the RG_LRU module.

        Args:
            input_size (int): The size of the input.
            hidden_size (int): The size of the hidden state.
        """
        super(RG_LRU, self).__init__()
        self.input_dim = input_dim
        hidden_dim = input_dim * mult
        self.hidden_dim = hidden_dim
        # Scalar-valued constant 'c'
        self.c = 8
        # Initialize weights and biases for the recurrence gate and input gate
        self.Wa = nn.Parameter(Tensor(hidden_dim, input_dim))
        self.Wx = nn.Parameter(Tensor(hidden_dim, input_dim))
        self.ba = nn.Parameter(Tensor(hidden_dim))
        self.bx = nn.Parameter(Tensor(hidden_dim))
        # Initialize the learnable parameter Λ for parameterizing 'a'
        self.Lambda = nn.Parameter(Tensor(hidden_dim))
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.kaiming_normal_(
            self.Wa, mode="fan_in", nonlinearity="sigmoid"
        )
        nn.init.kaiming_normal_(
            self.Wx, mode="fan_in", nonlinearity="sigmoid"
        )
        nn.init.constant_(self.ba, 0)
        nn.init.constant_(self.bx, 0)
        # Initialize Λ such that a is between 0.9 and 0.999
        self.Lambda.data.uniform_(
            torch.logit(torch.tensor(0.9)),
            torch.logit(torch.tensor(0.999)),
        )
    def forward(self, xt, ht_minus_1):
        """
        Performs a forward pass through the RG_LRU module.

        Args:
            xt (torch.Tensor): The input tensor.
            ht_minus_1 (torch.Tensor): The previous hidden state tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Compute the recurrence gate (rt)
        print(f"xt shape: {xt.shape}")
        print(f"ht_minus_1 shape: {ht_minus_1.shape}")
        rt = torch.sigmoid(F.linear(xt, self.Wa, self.ba))
        print(f"rt shape: {rt.shape}")
        # Compute the input gate (it)
        it = torch.sigmoid(F.linear(xt, self.Wx, self.bx))
        print(f"it shape: {it.shape}")
        # Compute 'a' in log space (log_a) and then raise it to the power of rt/self.c for numerical stability
        log_a = -F.softplus(
            -self.Lambda
        )  # Ensures the log(a) stays in a numerically stable range
        a = torch.exp(log_a * rt / self.c)
        print(f"a shape: {a.shape}")
        # Update hidden state 'ht'
        at = (
            a.diag_embed()
        )  # Make 'a' a diagonal matrix as recurrent weight 'a' in Eq. (4) is diagonal
        ht = at * ht_minus_1 + (1-a.pow(2)).sqrt() * (it * xt)
        # Output of the layer is 'yt', which is 'ht'
        yt = ht
        return yt


class RecurrentBlock(nn.Module):
    """
    A recurrent block module that applies a series of operations on the input tensor.

    This block is designed to process sequences by applying a combination of linear transformation,
    GELU activation, a temporal convolution, and a custom RG_LRU operation for enhanced feature extraction
    and temporal dynamics modeling.

    Args:
        input_dim (int): The number of input dimensions.
        hidden_dim (int): The number of hidden dimensions.

    Returns:
        torch.Tensor: The output tensor after applying the operations.
    """

    def __init__(self, input_dim, hidden_dim, rnn_width):
        super(RecurrentBlock, self).__init__()
        # Initialize RG_LRU component for recurrent gating with local recurrent units.
        self.rg_lru = RG_LRU(input_dim, hidden_dim)
        # Temporal convolution layer with kernel size 1 for dimensionality transformation without changing sequence length.
        self.temporal_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        # Linear transformation to map the hidden dimensions back to input dimensions.
        self.linear = nn.Linear(hidden_dim, input_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # Apply a linear transformation followed by GELU activation for non-linearity.
        print(f"x shape: {x.shape}")
        x_gel = self.linear(x)
        x_gel = F.gelu(x_gel)
        # Apply another linear transformation followed by temporal convolution for capturing temporal dynamics.
        x_rg = self.linear(x)
        x_rg = self.temporal_conv(x_rg.transpose(1, 2)).transpose(1, 2)
        # Apply the RG_LRU operation for recurrent gating and feature enhancement.
        x_rg = self.rg_lru(x_rg)
        # Element-wise multiplication of the GELU-activated and RG_LRU-processed tensors for feature fusion.
        combined_x = x_gel * x_rg
        # Final linear transformation to produce the output tensor.
        x = self.linear2(combined_x)
        return x


# Define a model that stacks these blocks
class GriffinModel(nn.Module):
    """
    GriffinModel is a PyTorch module that implements a deep neural network model.

    Args:
        input_dim (int): The dimensionality of the input.
        mlp_expansion_factor (int): The expansion factor for the hidden dimension in the MLP block.
        rnn_width (int): The width of the recurrent block.
        depth (int): The number of residual blocks in the model.

    Attributes:
        layers (nn.ModuleList): A list of ResidualBlock instances.
        gated_mlp (GatedMLPBlock): An instance of GatedMLPBlock.
        recurrent_block (RecurrentBlock): An instance of RecurrentBlock.

    """

    def __init__(
        self,
        vocab_size,
        input_dim=1024,
        mlp_expansion_factor=3,
        rnn_width=1536,
        depth=12,
    ):
        super(GriffinModel, self).__init__()
        self.hidden_dim = input_dim * mlp_expansion_factor  # Hidden dim
        self.input_dim = input_dim
        self.layers = nn.ModuleList(
            [ResidualBlock(input_dim, self.hidden_dim, rnn_width) for _ in range(depth)]
        )
        self.embd = Embedding(vocab_size, input_dim)

    def forward(self, x):
        x = self.embd(x)
        for layer in self.layers:
            x = layer(x) + x
        return output_head(x, self.input_dim)
