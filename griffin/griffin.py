"""Core Griffin-style language-model blocks.

This module implements the recurrent scaffold used by the repository:
RMS normalization, gated MLP residual blocks, a causal recurrent block,
an RG-LRU sequential scan, and a vocabulary projection head.
"""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square normalization with a learned per-channel scale."""

    def __init__(self, dim: int, eps: float = 1e-6):
        """Create an RMSNorm layer for the final dimension of an input tensor."""
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Normalize ``x`` across its final dimension."""
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.g


class GatedMLPBlock(nn.Module):
    """Gated feed-forward block used inside each residual layer."""

    def __init__(self, input_dim: int, hidden_dim: int):
        """Create the two input projections and output projection for the MLP."""
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(input_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply GELU(W1 x) * W2 x, then project back to ``input_dim``."""
        gate = F.gelu(self.linear1(x))
        return self.linear3(gate * self.linear2(x))


class RG_LRU(nn.Module):
    """Real-gated linear recurrent unit implemented as a sequential scan."""

    def __init__(self, rnn_width: int, c: float = 8.0):
        """Initialize recurrence gates, input gates, and diagonal decay parameters."""
        super().__init__()
        self.rnn_width = rnn_width
        self.c = c
        self.Wa = nn.Parameter(torch.empty(rnn_width, rnn_width))
        self.Wx = nn.Parameter(torch.empty(rnn_width, rnn_width))
        self.ba = nn.Parameter(torch.empty(rnn_width))
        self.bx = nn.Parameter(torch.empty(rnn_width))
        self.Lambda = nn.Parameter(torch.empty(rnn_width))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize RG-LRU parameters, including the paper-style decay range."""
        nn.init.xavier_uniform_(self.Wa)
        nn.init.xavier_uniform_(self.Wx)
        nn.init.zeros_(self.ba)
        nn.init.zeros_(self.bx)

        with torch.no_grad():
            # Sample a^c in the intended range, then recover the base a parameter.
            a_power_c = torch.empty_like(self.Lambda).uniform_(0.9, 0.999)
            base_a = a_power_c.pow(1.0 / self.c)
            self.Lambda.copy_(torch.logit(base_a))

    def forward(
        self,
        xt: Tensor,
        ht_minus_1: Tensor | None = None,
        *,
        return_state: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Run the recurrent update over a complete token sequence.

        Args:
            xt: Input activations with shape ``[batch, sequence, rnn_width]``.
            ht_minus_1: Optional initial recurrent state with shape
                ``[batch, rnn_width]``.
            return_state: When true, also return the final recurrent state.

        Returns:
            The sequence of recurrent states, plus the final state when requested.
        """
        if xt.dim() != 3:
            raise ValueError(f"RG_LRU expects [batch, sequence, width], got {tuple(xt.shape)}")

        batch_size, seq_len, width = xt.shape
        if width != self.rnn_width:
            raise ValueError(f"Expected width {self.rnn_width}, got {width}")

        if ht_minus_1 is None:
            h = xt.new_zeros(batch_size, width)
        else:
            if ht_minus_1.shape != (batch_size, width):
                raise ValueError(
                    "ht_minus_1 must have shape "
                    f"({batch_size}, {width}), got {tuple(ht_minus_1.shape)}"
                )
            h = ht_minus_1.to(device=xt.device, dtype=xt.dtype)

        recurrence_gate = torch.sigmoid(F.linear(xt, self.Wa, self.ba))
        input_gate = torch.sigmoid(F.linear(xt, self.Wx, self.bx))

        # The decay is diagonal, so the recurrence is elementwise. No D x D
        # diagonal tensor is needed.
        log_base_a = F.logsigmoid(self.Lambda).to(dtype=xt.dtype)
        log_a_t = self.c * recurrence_gate * log_base_a
        a_t = torch.exp(log_a_t)
        input_term = torch.sqrt(torch.clamp(1.0 - torch.exp(2.0 * log_a_t), min=0.0))
        input_term = input_term * input_gate * xt

        outputs = []
        for t in range(seq_len):
            # This scan is what carries token t into all later recurrent states.
            h = a_t[:, t] * h + input_term[:, t]
            outputs.append(h)

        yt = torch.stack(outputs, dim=1) if outputs else xt.new_empty(batch_size, 0, width)
        if return_state:
            return yt, h
        return yt


class RecurrentBlock(nn.Module):
    """Causal temporal-conv plus RG-LRU block with a gated output branch."""

    def __init__(self, input_dim: int, rnn_width: int, conv_kernel_size: int = 4):
        """Create independent branches for the recurrent and GELU paths."""
        super().__init__()
        self.conv_kernel_size = conv_kernel_size
        self.linear_x = nn.Linear(input_dim, rnn_width)
        self.linear_y = nn.Linear(input_dim, rnn_width)
        self.temporal_conv = nn.Conv1d(
            in_channels=rnn_width,
            out_channels=rnn_width,
            kernel_size=conv_kernel_size,
            padding=0,
            groups=rnn_width,
        )
        self.rg_lru = RG_LRU(rnn_width)
        self.linear_out = nn.Linear(rnn_width, input_dim)

    def forward(
        self,
        x: Tensor,
        ht_minus_1: Tensor | None = None,
        *,
        return_state: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Apply the causal recurrent block to ``[batch, sequence, input_dim]``."""
        y_branch = F.gelu(self.linear_y(x))

        x_branch = self.linear_x(x).transpose(1, 2)
        # Left padding makes the depthwise convolution autoregressive.
        x_branch = F.pad(x_branch, (self.conv_kernel_size - 1, 0))
        x_branch = self.temporal_conv(x_branch).transpose(1, 2)

        x_branch, new_state = self.rg_lru(x_branch, ht_minus_1, return_state=True)
        output = self.linear_out(x_branch * y_branch)

        if return_state:
            return output, new_state
        return output


class ResidualBlock(nn.Module):
    """One recurrent residual sublayer followed by one MLP residual sublayer."""

    def __init__(self, input_dim: int, expansion_factor: int, rnn_width: int):
        """Create the two pre-normalized residual sublayers."""
        super().__init__()
        hidden_dim = input_dim * expansion_factor
        self.norm1 = RMSNorm(input_dim)
        self.recurrent = RecurrentBlock(input_dim, rnn_width)
        self.norm2 = RMSNorm(input_dim)
        self.mlp = GatedMLPBlock(input_dim, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply recurrent and MLP residual updates without an extra outer skip."""
        recurrent_residual = x
        recurrent_update = self.recurrent(self.norm1(x))
        x = recurrent_residual + recurrent_update

        mlp_residual = x
        mlp_update = self.mlp(self.norm2(x))
        x = mlp_residual + mlp_update
        return x


class GriffinModel(nn.Module):
    """Stacked recurrent language model that returns raw vocabulary logits."""

    def __init__(
        self,
        vocab_size: int,
        input_dim: int = 1024,
        mlp_expansion_factor: int = 3,
        rnn_width: int = 1536,
        depth: int = 12,
        *,
        tie_embeddings: bool = False,
    ):
        """Create embeddings, residual layers, final norm, and LM head."""
        super().__init__()
        self.vocab_size = vocab_size
        self.input_dim = input_dim
        self.embd = nn.Embedding(vocab_size, input_dim)
        self.layers = nn.ModuleList(
            [
                ResidualBlock(input_dim, mlp_expansion_factor, rnn_width)
                for _ in range(depth)
            ]
        )
        self.final_norm = RMSNorm(input_dim)
        self.lm_head = nn.Linear(input_dim, vocab_size, bias=False)

        if tie_embeddings:
            self.lm_head.weight = self.embd.weight

    def forward(self, token_ids: Tensor) -> Tensor:
        """Embed token IDs and return unnormalized ``[B, T, vocab_size]`` logits."""
        x = self.embd(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return self.lm_head(x)
