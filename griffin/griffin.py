from __future__ import annotations

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.g


class GatedMLPBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(input_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: Tensor) -> Tensor:
        gate = F.gelu(self.linear1(x))
        return self.linear3(gate * self.linear2(x))


class RG_LRU(nn.Module):
    def __init__(self, rnn_width: int, c: float = 8.0):
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
        nn.init.xavier_uniform_(self.Wa)
        nn.init.xavier_uniform_(self.Wx)
        nn.init.zeros_(self.ba)
        nn.init.zeros_(self.bx)

        with torch.no_grad():
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

        log_base_a = F.logsigmoid(self.Lambda).to(dtype=xt.dtype)
        log_a_t = self.c * recurrence_gate * log_base_a
        a_t = torch.exp(log_a_t)
        input_term = torch.sqrt(torch.clamp(1.0 - torch.exp(2.0 * log_a_t), min=0.0))
        input_term = input_term * input_gate * xt

        outputs = []
        for t in range(seq_len):
            h = a_t[:, t] * h + input_term[:, t]
            outputs.append(h)

        yt = torch.stack(outputs, dim=1) if outputs else xt.new_empty(batch_size, 0, width)
        if return_state:
            return yt, h
        return yt


class RecurrentBlock(nn.Module):
    def __init__(self, input_dim: int, rnn_width: int, conv_kernel_size: int = 4):
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
        y_branch = F.gelu(self.linear_y(x))

        x_branch = self.linear_x(x).transpose(1, 2)
        x_branch = F.pad(x_branch, (self.conv_kernel_size - 1, 0))
        x_branch = self.temporal_conv(x_branch).transpose(1, 2)

        x_branch, new_state = self.rg_lru(x_branch, ht_minus_1, return_state=True)
        output = self.linear_out(x_branch * y_branch)

        if return_state:
            return output, new_state
        return output


class ResidualBlock(nn.Module):
    def __init__(self, input_dim: int, expansion_factor: int, rnn_width: int):
        super().__init__()
        hidden_dim = input_dim * expansion_factor
        self.norm1 = RMSNorm(input_dim)
        self.recurrent = RecurrentBlock(input_dim, rnn_width)
        self.norm2 = RMSNorm(input_dim)
        self.mlp = GatedMLPBlock(input_dim, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.recurrent(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GriffinModel(nn.Module):
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
        x = self.embd(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return self.lm_head(x)
