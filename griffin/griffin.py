"""Griffin language-model blocks with recurrent and local-attention mixing.

The implementation follows the architecture described in the Griffin paper:
pre-normalized residual blocks, gated MLPs, RG-LRU recurrent blocks, and a
repeating recurrent/recurrent/local-attention temporal-mixing schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence, TypeAlias, overload

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


TemporalBlockType = Literal["recurrent", "attention"]


@dataclass
class RecurrentBlockCache:
    """State needed to continue a recurrent block on a later token chunk."""

    rnn_state: Tensor
    conv_state: Tensor


@dataclass
class AttentionBlockCache:
    """Bounded local-attention KV cache and the next absolute position."""

    keys: Tensor
    values: Tensor
    position: int


LayerCache: TypeAlias = RecurrentBlockCache | AttentionBlockCache


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


def _compatible_block_count(width: int, requested_blocks: int) -> int:
    """Return the largest requested-or-smaller gate block count dividing width."""
    if width <= 0 or requested_blocks <= 0:
        raise ValueError("width and requested_blocks must be positive")
    for block_count in range(min(width, requested_blocks), 0, -1):
        if width % block_count == 0:
            return block_count
    raise AssertionError("Every positive width is divisible by one")


class BlockDiagonalLinear(nn.Module):
    """Linear projection whose weight contains independent diagonal blocks."""

    def __init__(self, width: int, num_blocks: int = 16):
        """Create a block-diagonal projection and zero-initialized bias."""
        super().__init__()
        self.width = width
        self.num_blocks = _compatible_block_count(width, num_blocks)
        self.block_width = width // self.num_blocks
        self.weight = nn.Parameter(
            torch.empty(self.num_blocks, self.block_width, self.block_width)
        )
        self.bias = nn.Parameter(torch.empty(self.num_blocks, self.block_width))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Apply LeCun-normal initialization independently to every block."""
        nn.init.normal_(self.weight, mean=0.0, std=math.sqrt(1.0 / self.block_width))
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Project the last dimension of ``x`` without materializing a dense matrix."""
        blocked = x.reshape(*x.shape[:-1], self.num_blocks, self.block_width)
        projected = torch.einsum("...bi,boi->...bo", blocked, self.weight)
        projected = projected + self.bias
        return projected.reshape(*x.shape[:-1], self.width)


class RG_LRU(nn.Module):
    """Real-gated linear recurrent unit implemented as a sequential scan."""

    def __init__(self, rnn_width: int, c: float = 8.0, gate_blocks: int = 16):
        """Initialize block-diagonal gates and diagonal decay parameters."""
        super().__init__()
        self.rnn_width = rnn_width
        self.c = c
        self.recurrence_gate = BlockDiagonalLinear(rnn_width, gate_blocks)
        self.input_gate = BlockDiagonalLinear(rnn_width, gate_blocks)
        self.Lambda = nn.Parameter(torch.empty(rnn_width))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize gates with LeCun normal and ``a^c`` uniformly in paper range."""
        self.recurrence_gate.reset_parameters()
        self.input_gate.reset_parameters()
        with torch.no_grad():
            # This sign convention follows a = sigmoid(Lambda). It is equivalent
            # to the paper appendix convention after replacing Lambda by -Lambda.
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
        """Run the recurrent update over ``[batch, sequence, rnn_width]`` inputs."""
        if xt.dim() != 3:
            raise ValueError(f"RG_LRU expects [batch, sequence, width], got {tuple(xt.shape)}")

        batch_size, seq_len, width = xt.shape
        if width != self.rnn_width:
            raise ValueError(f"Expected width {self.rnn_width}, got {width}")

        # Float32 accumulation protects long scans when activations use bf16/fp16.
        accumulation_dtype = torch.float32 if xt.dtype in (torch.float16, torch.bfloat16) else xt.dtype
        if ht_minus_1 is None:
            h = torch.zeros(batch_size, width, device=xt.device, dtype=accumulation_dtype)
        else:
            if ht_minus_1.shape != (batch_size, width):
                raise ValueError(
                    "ht_minus_1 must have shape "
                    f"({batch_size}, {width}), got {tuple(ht_minus_1.shape)}"
                )
            h = ht_minus_1.to(device=xt.device, dtype=accumulation_dtype)

        recurrence_gate = torch.sigmoid(self.recurrence_gate(xt))
        input_gate = torch.sigmoid(self.input_gate(xt))
        log_base_a = F.logsigmoid(self.Lambda).to(dtype=xt.dtype)
        log_a_t = self.c * recurrence_gate * log_base_a
        a_t = torch.exp(log_a_t)

        # -expm1 is accurate when a_t is close to one, where 1 - exp(.) cancels.
        gamma = torch.sqrt(torch.clamp(-torch.expm1(2.0 * log_a_t), min=0.0))
        normalized_input = gamma * input_gate * xt

        outputs = []
        for t in range(seq_len):
            h = a_t[:, t].to(accumulation_dtype) * h
            h = h + normalized_input[:, t].to(accumulation_dtype)
            outputs.append(h.to(xt.dtype))

        yt = torch.stack(outputs, dim=1) if outputs else xt.new_empty(batch_size, 0, width)
        if return_state:
            return yt, h
        return yt


class RecurrentBlock(nn.Module):
    """Causal temporal-convolution plus RG-LRU block with cached decoding."""

    def __init__(
        self,
        input_dim: int,
        rnn_width: int,
        conv_kernel_size: int = 4,
        gate_blocks: int = 16,
    ):
        """Create independent recurrent and GELU branches."""
        super().__init__()
        if conv_kernel_size <= 0:
            raise ValueError("conv_kernel_size must be positive")
        self.rnn_width = rnn_width
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
        self.rg_lru = RG_LRU(rnn_width, gate_blocks=gate_blocks)
        self.linear_out = nn.Linear(rnn_width, input_dim)

    def _empty_conv_state(self, x: Tensor) -> Tensor:
        """Create the zero history used at the beginning of a sequence."""
        return x.new_zeros(x.size(0), self.conv_kernel_size - 1, self.rnn_width)

    @overload
    def forward(
        self, x: Tensor, cache: RecurrentBlockCache | None = None, *, return_cache: Literal[False] = False
    ) -> Tensor:
        """Apply the block without returning an updated cache."""
        ...

    @overload
    def forward(
        self, x: Tensor, cache: RecurrentBlockCache | None = None, *, return_cache: Literal[True]
    ) -> tuple[Tensor, RecurrentBlockCache]:
        """Apply the block and return its updated recurrent cache."""
        ...

    def forward(
        self,
        x: Tensor,
        cache: RecurrentBlockCache | None = None,
        *,
        return_cache: bool = False,
    ) -> Tensor | tuple[Tensor, RecurrentBlockCache]:
        """Apply the block and optionally return convolution and recurrent state."""
        if x.dim() != 3:
            raise ValueError(f"RecurrentBlock expects [batch, sequence, width], got {tuple(x.shape)}")

        y_branch = F.gelu(self.linear_y(x))
        projected_x = self.linear_x(x)
        conv_state = self._empty_conv_state(projected_x) if cache is None else cache.conv_state
        if conv_state.shape != (
            x.size(0),
            self.conv_kernel_size - 1,
            self.rnn_width,
        ):
            raise ValueError("Recurrent convolution cache has an incompatible shape")

        # Prepending cached projected inputs makes chunked and full convolution identical.
        conv_input = torch.cat([conv_state.to(projected_x), projected_x], dim=1)
        x_branch = self.temporal_conv(conv_input.transpose(1, 2)).transpose(1, 2)
        rnn_state = None if cache is None else cache.rnn_state
        x_branch, new_rnn_state = self.rg_lru(x_branch, rnn_state, return_state=True)
        output = self.linear_out(x_branch * y_branch)

        if return_cache:
            history = self.conv_kernel_size - 1
            new_conv_state = conv_input[:, -history:] if history else conv_input[:, :0]
            return output, RecurrentBlockCache(new_rnn_state, new_conv_state)
        return output


def _rotate_half(x: Tensor) -> Tensor:
    """Rotate pairs in the final dimension for rotary position embeddings."""
    first, second = x.chunk(2, dim=-1)
    return torch.cat([-second, first], dim=-1)


def _apply_rope(x: Tensor, positions: Tensor) -> Tensor:
    """Apply rotary position embeddings to ``[B, T, H, head_dim]`` tensors."""
    head_dim = x.size(-1)
    if head_dim % 2:
        raise ValueError("Rotary head dimension must be even")
    frequency = torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32)
    inverse_frequency = 1.0 / (10000.0 ** (frequency / head_dim))
    angles = torch.outer(positions.to(torch.float32), inverse_frequency)
    angles = torch.cat([angles, angles], dim=-1).to(dtype=x.dtype)
    cos = angles.cos()[None, :, None, :]
    sin = angles.sin()[None, :, None, :]
    return x * cos + _rotate_half(x) * sin


class LocalMQAAttention(nn.Module):
    """Causal sliding-window multi-query attention with RoPE and KV caching."""

    def __init__(self, input_dim: int, num_heads: int, window_size: int = 1024):
        """Create query, shared key/value, and output projections."""
        super().__init__()
        if input_dim % num_heads:
            raise ValueError("input_dim must be divisible by num_heads")
        if (input_dim // num_heads) % 2:
            raise ValueError("attention head dimension must be even for RoPE")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.window_size = window_size
        self.query = nn.Linear(input_dim, input_dim, bias=False)
        self.key = nn.Linear(input_dim, self.head_dim, bias=False)
        self.value = nn.Linear(input_dim, self.head_dim, bias=False)
        self.output = nn.Linear(input_dim, input_dim, bias=False)

    @overload
    def forward(
        self, x: Tensor, cache: AttentionBlockCache | None = None, *, return_cache: Literal[False] = False
    ) -> Tensor:
        """Apply local attention without returning an updated cache."""
        ...

    @overload
    def forward(
        self, x: Tensor, cache: AttentionBlockCache | None = None, *, return_cache: Literal[True]
    ) -> tuple[Tensor, AttentionBlockCache]:
        """Apply local attention and return its updated KV cache."""
        ...

    def forward(
        self,
        x: Tensor,
        cache: AttentionBlockCache | None = None,
        *,
        return_cache: bool = False,
    ) -> Tensor | tuple[Tensor, AttentionBlockCache]:
        """Apply local MQA and optionally return a bounded cache for later chunks."""
        batch_size, seq_len, _ = x.shape
        position = 0 if cache is None else cache.position
        positions = torch.arange(position, position + seq_len, device=x.device)

        queries = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        queries = _apply_rope(queries, positions)
        keys = self.key(x).view(batch_size, seq_len, 1, self.head_dim)
        keys = _apply_rope(keys, positions).squeeze(2)
        values = self.value(x)

        if cache is not None:
            keys = torch.cat([cache.keys.to(keys), keys], dim=1)
            values = torch.cat([cache.values.to(values), values], dim=1)

        past_len = keys.size(1) - seq_len
        key_positions = torch.arange(position - past_len, position + seq_len, device=x.device)
        query_positions = positions[:, None]
        causal = key_positions[None, :] <= query_positions
        local = key_positions[None, :] >= query_positions - self.window_size + 1
        allowed = causal & local

        scores = torch.einsum("bthd,bsd->bhts", queries, keys)
        scores = scores * (self.head_dim ** -0.5)
        scores = scores.masked_fill(~allowed[None, None, :, :], float("-inf"))
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
        attended = torch.einsum("bhts,bsd->bthd", weights, values)
        output = self.output(attended.reshape(batch_size, seq_len, self.input_dim))

        if return_cache:
            cache_length = min(self.window_size - 1, keys.size(1))
            if cache_length:
                cached_keys = keys[:, -cache_length:]
                cached_values = values[:, -cache_length:]
            else:
                cached_keys = keys[:, :0]
                cached_values = values[:, :0]
            return output, AttentionBlockCache(cached_keys, cached_values, position + seq_len)
        return output


class ResidualBlock(nn.Module):
    """One temporal-mixing residual sublayer followed by a gated MLP."""

    def __init__(
        self,
        input_dim: int,
        expansion_factor: int,
        rnn_width: int,
        *,
        temporal_block_type: TemporalBlockType = "recurrent",
        attention_heads: int = 1,
        attention_window_size: int = 1024,
        gate_blocks: int = 16,
    ):
        """Create a pre-normalized recurrent or attention residual block."""
        super().__init__()
        hidden_dim = input_dim * expansion_factor
        self.temporal_block_type = temporal_block_type
        self.norm1 = RMSNorm(input_dim)
        if temporal_block_type == "recurrent":
            self.recurrent = RecurrentBlock(input_dim, rnn_width, gate_blocks=gate_blocks)
            self.temporal = self.recurrent
        elif temporal_block_type == "attention":
            self.attention = LocalMQAAttention(
                input_dim,
                attention_heads,
                attention_window_size,
            )
            self.temporal = self.attention
        else:
            raise ValueError(f"Unknown temporal block type: {temporal_block_type}")
        self.norm2 = RMSNorm(input_dim)
        self.mlp = GatedMLPBlock(input_dim, hidden_dim)

    def forward(
        self,
        x: Tensor,
        cache: LayerCache | None = None,
        *,
        return_cache: bool = False,
    ) -> Tensor | tuple[Tensor, LayerCache]:
        """Apply temporal and MLP residual updates, optionally returning layer cache."""
        temporal_result = self.temporal(self.norm1(x), cache, return_cache=return_cache)
        if return_cache:
            temporal_update, new_cache = temporal_result
        else:
            temporal_update = temporal_result
            new_cache = None
        x = x + temporal_update
        x = x + self.mlp(self.norm2(x))
        if return_cache:
            return x, new_cache
        return x


def _default_attention_heads(input_dim: int) -> int:
    """Choose paper-style 128-wide heads when possible, otherwise a valid small head."""
    preferred = max(1, input_dim // 128)
    for num_heads in range(preferred, 0, -1):
        if input_dim % num_heads == 0 and (input_dim // num_heads) % 2 == 0:
            return num_heads
    raise ValueError("input_dim must permit an even attention head dimension")


def _griffin_schedule(depth: int) -> tuple[TemporalBlockType, ...]:
    """Build the paper's repeating recurrent, recurrent, attention schedule."""
    pattern: tuple[TemporalBlockType, ...] = ("recurrent", "recurrent", "attention")
    return tuple(pattern[index % len(pattern)] for index in range(depth))


class GriffinModel(nn.Module):
    """Hybrid Griffin language model returning raw vocabulary logits."""

    def __init__(
        self,
        vocab_size: int,
        input_dim: int = 1024,
        mlp_expansion_factor: int = 3,
        rnn_width: int = 1536,
        depth: int = 12,
        *,
        attention_heads: int | None = None,
        attention_window_size: int = 1024,
        block_types: Sequence[TemporalBlockType] | None = None,
        gate_blocks: int = 16,
        tie_embeddings: bool = True,
    ):
        """Create embeddings, hybrid residual layers, final norm, and tied LM head."""
        super().__init__()
        self.vocab_size = vocab_size
        self.input_dim = input_dim
        self.block_types = tuple(block_types) if block_types is not None else _griffin_schedule(depth)
        if len(self.block_types) != depth:
            raise ValueError("block_types must contain exactly depth entries")
        attention_heads = attention_heads or _default_attention_heads(input_dim)

        self.embd = nn.Embedding(vocab_size, input_dim)
        nn.init.normal_(self.embd.weight, mean=0.0, std=math.sqrt(1.0 / input_dim))
        self.embedding_scale = math.sqrt(input_dim)
        self.layers = nn.ModuleList(
            [
                ResidualBlock(
                    input_dim,
                    mlp_expansion_factor,
                    rnn_width,
                    temporal_block_type=block_type,
                    attention_heads=attention_heads,
                    attention_window_size=attention_window_size,
                    gate_blocks=gate_blocks,
                )
                for block_type in self.block_types
            ]
        )
        self.final_norm = RMSNorm(input_dim)
        self.lm_head = nn.Linear(input_dim, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.embd.weight

    @overload
    def forward(
        self, token_ids: Tensor, cache: Sequence[LayerCache] | None = None, *, return_cache: Literal[False] = False
    ) -> Tensor:
        """Return logits without constructing updated layer caches."""
        ...

    @overload
    def forward(
        self, token_ids: Tensor, cache: Sequence[LayerCache] | None = None, *, return_cache: Literal[True]
    ) -> tuple[Tensor, list[LayerCache]]:
        """Return logits and one updated cache per residual layer."""
        ...

    def forward(
        self,
        token_ids: Tensor,
        cache: Sequence[LayerCache] | None = None,
        *,
        return_cache: bool = False,
    ) -> Tensor | tuple[Tensor, list[LayerCache]]:
        """Embed token IDs and return logits, optionally with bounded decode caches."""
        if cache is not None and len(cache) != len(self.layers):
            raise ValueError("cache must contain one entry per residual layer")

        # Scaling restores unit-variance input activations after LeCun embedding init.
        x = self.embd(token_ids) * self.embedding_scale
        new_cache: list[LayerCache] = []
        for index, layer in enumerate(self.layers):
            layer_cache = None if cache is None else cache[index]
            if return_cache:
                x, updated_cache = layer(x, layer_cache, return_cache=True)
                new_cache.append(updated_cache)
            else:
                x = layer(x, layer_cache)
        logits = self.lm_head(self.final_norm(x))
        if return_cache:
            return logits, new_cache
        return logits
