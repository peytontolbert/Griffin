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
ScanMode = Literal["auto", "sequential", "associative"]


@dataclass
class RecurrentBlockCache:
    """State needed to continue a recurrent block on a later token chunk."""

    rnn_state: Tensor
    conv_state: Tensor
    segment_position: Tensor


@dataclass
class AttentionBlockCache:
    """Bounded local-attention KV cache and last per-sequence position."""

    keys: Tensor
    values: Tensor
    valid_mask: Tensor
    segment_position: Tensor


LayerCache: TypeAlias = RecurrentBlockCache | AttentionBlockCache


class _SqrtBoundDerivative(torch.autograd.Function):
    """Square root whose backward derivative is bounded near zero."""

    max_gradient = 1000.0

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        """Compute the ordinary square root and retain its input for backward."""
        ctx.save_for_backward(x)
        return torch.sqrt(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor]:
        """Apply the analytical derivative with a finite upper bound."""
        (x,) = ctx.saved_tensors
        minimum = 1.0 / (_SqrtBoundDerivative.max_gradient**2)
        denominator = torch.sqrt(torch.clamp(4.0 * x, min=minimum))
        return (grad_output / denominator,)


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

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        final_w_init_variance_scale: float = 1.0,
    ):
        """Create the two input projections and output projection for the MLP."""
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(input_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, input_dim)
        self.final_w_init_variance_scale = final_w_init_variance_scale
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize input projections with LeCun normal and scale the residual output."""
        input_std = math.sqrt(1.0 / self.linear1.in_features)
        output_std = math.sqrt(
            self.final_w_init_variance_scale / self.linear3.in_features
        )
        nn.init.normal_(self.linear1.weight, mean=0.0, std=input_std)
        nn.init.normal_(self.linear2.weight, mean=0.0, std=input_std)
        nn.init.normal_(self.linear3.weight, mean=0.0, std=output_std)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)
        nn.init.zeros_(self.linear3.bias)

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
    """Real-gated linear recurrent unit with sequential and parallel scan paths."""

    def __init__(
        self,
        rnn_width: int,
        c: float = 8.0,
        gate_blocks: int = 16,
        scan_mode: ScanMode = "auto",
    ):
        """Initialize block-diagonal gates and diagonal decay parameters."""
        super().__init__()
        self.rnn_width = rnn_width
        self.c = c
        if scan_mode not in ("auto", "sequential", "associative"):
            raise ValueError(f"Unknown RG-LRU scan mode: {scan_mode}")
        self.scan_mode = scan_mode
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
        reset_mask: Tensor | None = None,
        active_mask: Tensor | None = None,
        return_state: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Run the recurrent update, resetting state where ``reset_mask`` is true."""
        if xt.dim() != 3:
            raise ValueError(f"RG_LRU expects [batch, sequence, width], got {tuple(xt.shape)}")

        batch_size, seq_len, width = xt.shape
        if width != self.rnn_width:
            raise ValueError(f"Expected width {self.rnn_width}, got {width}")
        if reset_mask is None:
            reset_mask = torch.zeros(batch_size, seq_len, device=xt.device, dtype=torch.bool)
        elif reset_mask.shape != (batch_size, seq_len):
            raise ValueError(
                "reset_mask must have shape "
                f"({batch_size}, {seq_len}), got {tuple(reset_mask.shape)}"
            )
        else:
            reset_mask = reset_mask.to(device=xt.device, dtype=torch.bool)
        if active_mask is None:
            active_mask = torch.ones(batch_size, seq_len, device=xt.device, dtype=torch.bool)
        elif active_mask.shape != (batch_size, seq_len):
            raise ValueError(
                "active_mask must have shape "
                f"({batch_size}, {seq_len}), got {tuple(active_mask.shape)}"
            )
        else:
            active_mask = active_mask.to(device=xt.device, dtype=torch.bool)

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
        gamma_squared = torch.clamp(-torch.expm1(2.0 * log_a_t), min=0.0)
        gamma = _SqrtBoundDerivative.apply(gamma_squared)
        # At a document boundary the reference recurrence discards history and
        # applies the gated input without stationary-state normalization.
        a_t = torch.where(reset_mask[..., None], torch.zeros_like(a_t), a_t)
        gamma = torch.where(reset_mask[..., None], torch.ones_like(gamma), gamma)
        normalized_input = gamma * input_gate * xt
        # Padding is the identity affine transform, so it cannot advance state.
        a_t = torch.where(active_mask[..., None], a_t, torch.ones_like(a_t))
        normalized_input = torch.where(
            active_mask[..., None], normalized_input, torch.zeros_like(normalized_input)
        )

        a_scan = a_t.to(accumulation_dtype)
        x_scan = normalized_input.to(accumulation_dtype)
        use_associative = self.scan_mode == "associative" or (
            self.scan_mode == "auto" and self.training and xt.is_cuda and seq_len > 1
        )
        if use_associative:
            yt_accumulated = self._associative_scan(a_scan, x_scan, h)
            h = yt_accumulated[:, -1]
        else:
            outputs = []
            for t in range(seq_len):
                h = a_scan[:, t] * h + x_scan[:, t]
                outputs.append(h)
            yt_accumulated = torch.stack(outputs, dim=1)

        yt = yt_accumulated.to(xt.dtype)
        if return_state:
            return yt, h
        return yt

    @staticmethod
    def _associative_scan(a: Tensor, x: Tensor, initial_state: Tensor) -> Tensor:
        """Evaluate ``h[t] = a[t] * h[t-1] + x[t]`` by affine pair composition."""
        # Hillis-Steele composition has O(log T) dependency depth. It performs
        # more total work than a fused linear scan, but exposes token parallelism
        # without relying on PyTorch's private, CUDA-only prototype scan API.
        offset = 1
        while offset < a.size(1):
            previous_a = a[:, :-offset]
            current_a = a[:, offset:]
            composed_x = x[:, offset:] + current_a * x[:, :-offset]
            composed_a = current_a * previous_a
            a = torch.cat([a[:, :offset], composed_a], dim=1)
            x = torch.cat([x[:, :offset], composed_x], dim=1)
            offset *= 2
        return x + a * initial_state[:, None, :]


def _resolve_segment_positions(
    x: Tensor,
    segment_pos: Tensor | None,
    previous_position: Tensor | None = None,
    active_mask: Tensor | None = None,
) -> Tensor:
    """Validate explicit positions or create contiguous per-sequence positions."""
    batch_size, seq_len = x.shape[:2]
    if seq_len == 0:
        raise ValueError("Empty token sequences are not supported")
    if segment_pos is not None:
        if segment_pos.shape != (batch_size, seq_len):
            raise ValueError(
                "segment_pos must have shape "
                f"({batch_size}, {seq_len}), got {tuple(segment_pos.shape)}"
            )
        return segment_pos.to(device=x.device, dtype=torch.long)

    if active_mask is None:
        active_mask = torch.ones(batch_size, seq_len, device=x.device, dtype=torch.bool)
    if previous_position is None:
        previous_position = torch.full(
            (batch_size,), -1, device=x.device, dtype=torch.long
        )
    else:
        if previous_position.shape != (batch_size,):
            raise ValueError("Cached segment positions have an incompatible shape")
        previous_position = previous_position.to(device=x.device, dtype=torch.long)
    increments = active_mask.to(torch.long).cumsum(dim=1)
    return previous_position[:, None] + increments


def _last_active_position(
    segment_pos: Tensor,
    active_mask: Tensor,
    previous_position: Tensor | None,
) -> Tensor:
    """Return each batch item's final valid position without advancing on padding."""
    batch_size = segment_pos.size(0)
    if previous_position is None:
        result = torch.full(
            (batch_size,), -1, device=segment_pos.device, dtype=torch.long
        )
    else:
        result = previous_position.to(device=segment_pos.device, dtype=torch.long).clone()
    for batch_index in range(batch_size):
        valid_positions = segment_pos[batch_index, active_mask[batch_index]]
        if valid_positions.numel():
            result[batch_index] = valid_positions[-1]
    return result


class RecurrentBlock(nn.Module):
    """Causal temporal-convolution plus RG-LRU block with cached decoding."""

    def __init__(
        self,
        input_dim: int,
        rnn_width: int,
        conv_kernel_size: int = 4,
        gate_blocks: int = 16,
        final_w_init_variance_scale: float = 1.0,
        scan_mode: ScanMode = "auto",
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
        self.rg_lru = RG_LRU(rnn_width, gate_blocks=gate_blocks, scan_mode=scan_mode)
        self.linear_out = nn.Linear(rnn_width, input_dim)
        self.final_w_init_variance_scale = final_w_init_variance_scale
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Apply the reference initialization to projections and depthwise convolution."""
        input_std = math.sqrt(1.0 / self.linear_x.in_features)
        output_std = math.sqrt(
            self.final_w_init_variance_scale / self.linear_out.in_features
        )
        convolution_std = math.sqrt(0.01 / self.conv_kernel_size)
        nn.init.normal_(self.linear_x.weight, mean=0.0, std=input_std)
        nn.init.normal_(self.linear_y.weight, mean=0.0, std=input_std)
        nn.init.normal_(self.linear_out.weight, mean=0.0, std=output_std)
        nn.init.normal_(self.temporal_conv.weight, mean=0.0, std=convolution_std)
        nn.init.zeros_(self.linear_x.bias)
        nn.init.zeros_(self.linear_y.bias)
        nn.init.zeros_(self.linear_out.bias)
        nn.init.zeros_(self.temporal_conv.bias)

    def _empty_conv_state(self, x: Tensor) -> Tensor:
        """Create the zero history used at the beginning of a sequence."""
        return x.new_zeros(x.size(0), self.conv_kernel_size - 1, self.rnn_width)

    @overload
    def forward(
        self,
        x: Tensor,
        cache: RecurrentBlockCache | None = None,
        segment_pos: Tensor | None = None,
        *,
        active_mask: Tensor | None = None,
        return_cache: Literal[False] = False,
    ) -> Tensor:
        """Apply the block without returning an updated cache."""
        ...

    @overload
    def forward(
        self,
        x: Tensor,
        cache: RecurrentBlockCache | None = None,
        segment_pos: Tensor | None = None,
        *,
        active_mask: Tensor | None = None,
        return_cache: Literal[True],
    ) -> tuple[Tensor, RecurrentBlockCache]:
        """Apply the block and return its updated recurrent cache."""
        ...

    def forward(
        self,
        x: Tensor,
        cache: RecurrentBlockCache | None = None,
        segment_pos: Tensor | None = None,
        *,
        active_mask: Tensor | None = None,
        return_cache: bool = False,
    ) -> Tensor | tuple[Tensor, RecurrentBlockCache]:
        """Apply the block and optionally return convolution and recurrent state."""
        if x.dim() != 3:
            raise ValueError(f"RecurrentBlock expects [batch, sequence, width], got {tuple(x.shape)}")
        batch_size, seq_len = x.shape[:2]
        all_tokens_active = active_mask is None
        if active_mask is None:
            active_mask = torch.ones(batch_size, seq_len, device=x.device, dtype=torch.bool)
        elif active_mask.shape != (batch_size, seq_len):
            raise ValueError("active_mask must match the input [batch, sequence] dimensions")
        else:
            active_mask = active_mask.to(device=x.device, dtype=torch.bool)
        previous_position = None if cache is None else cache.segment_position
        segment_pos = _resolve_segment_positions(
            x, segment_pos, previous_position, active_mask
        )

        y_branch = F.gelu(self.linear_y(x))
        projected_x = self.linear_x(x)
        conv_state = self._empty_conv_state(projected_x) if cache is None else cache.conv_state
        if conv_state.shape != (
            x.size(0),
            self.conv_kernel_size - 1,
            self.rnn_width,
        ):
            raise ValueError("Recurrent convolution cache has an incompatible shape")

        history = self.conv_kernel_size - 1
        conv_weight = self.temporal_conv.weight[:, 0, :]
        if all_tokens_active or bool(active_mask.all()):
            # The dense path keeps training efficient when no right padding is present.
            conv_input = torch.cat([conv_state.to(projected_x), projected_x], dim=1)
            x_branch = projected_x.new_zeros(projected_x.shape)
            if self.temporal_conv.bias is not None:
                x_branch = x_branch + self.temporal_conv.bias[None, None, :]
            for lag in range(self.conv_kernel_size):
                source = conv_input[:, history - lag : history - lag + seq_len]
                weight = conv_weight[:, self.conv_kernel_size - lag - 1]
                same_document = (segment_pos >= lag).to(dtype=source.dtype)
                x_branch = x_branch + source * same_document[..., None] * weight[None, None, :]
            new_conv_state = conv_input[:, -history:] if history else conv_input[:, :0]
        else:
            # Padding must not enter convolution history, so update the short
            # depthwise state explicitly for variable-length prompt batches.
            running_state = conv_state.to(projected_x)
            convolved = []
            for position in range(seq_len):
                active = active_mask[:, position, None]
                reset = (segment_pos[:, position] == 0)[:, None, None] & active[:, :, None]
                running_state = torch.where(reset, torch.zeros_like(running_state), running_state)
                current = projected_x[:, position]
                if history:
                    value = torch.einsum("bkr,rk->br", running_state, conv_weight[:, :history])
                else:
                    value = torch.zeros_like(current)
                value = value + current * conv_weight[:, -1]
                if self.temporal_conv.bias is not None:
                    value = value + self.temporal_conv.bias
                convolved.append(torch.where(active, value, torch.zeros_like(value)))
                if history:
                    shifted = torch.cat([running_state[:, 1:], current[:, None, :]], dim=1)
                    running_state = torch.where(active[:, :, None], shifted, running_state)
            x_branch = torch.stack(convolved, dim=1)
            new_conv_state = running_state

        rnn_state = None if cache is None else cache.rnn_state
        x_branch, new_rnn_state = self.rg_lru(
            x_branch,
            rnn_state,
            reset_mask=segment_pos == 0,
            active_mask=active_mask,
            return_state=True,
        )
        output = self.linear_out(x_branch * y_branch)
        output = torch.where(active_mask[..., None], output, torch.zeros_like(output))

        if return_cache:
            return output, RecurrentBlockCache(
                new_rnn_state,
                new_conv_state,
                _last_active_position(segment_pos, active_mask, previous_position),
            )
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
    if positions.dim() == 1:
        angles = torch.outer(positions.to(torch.float32), inverse_frequency)[None, :, :]
    elif positions.dim() == 2:
        angles = positions.to(torch.float32)[..., None] * inverse_frequency[None, None, :]
    else:
        raise ValueError("RoPE positions must have shape [T] or [B, T]")
    angles = torch.cat([angles, angles], dim=-1).to(dtype=x.dtype)
    cos = angles.cos()[:, :, None, :]
    sin = angles.sin()[:, :, None, :]
    return x * cos + _rotate_half(x) * sin


class LocalMQAAttention(nn.Module):
    """Causal sliding-window multi-query attention with RoPE and KV caching."""

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        window_size: int = 1024,
        final_w_init_variance_scale: float = 1.0,
        chunk_size: int = 128,
    ):
        """Create query, shared key/value, and output projections."""
        super().__init__()
        if input_dim % num_heads:
            raise ValueError("input_dim must be divisible by num_heads")
        if (input_dim // num_heads) % 2:
            raise ValueError("attention head dimension must be even for RoPE")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.final_w_init_variance_scale = final_w_init_variance_scale
        self.query = nn.Linear(input_dim, input_dim, bias=False)
        self.key = nn.Linear(input_dim, self.head_dim, bias=False)
        self.value = nn.Linear(input_dim, self.head_dim, bias=False)
        self.output = nn.Linear(input_dim, input_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Apply LeCun-normal input and depth-scaled residual-output initialization."""
        input_std = math.sqrt(1.0 / self.input_dim)
        output_std = math.sqrt(
            self.final_w_init_variance_scale / self.input_dim
        )
        nn.init.normal_(self.query.weight, mean=0.0, std=input_std)
        nn.init.normal_(self.key.weight, mean=0.0, std=input_std)
        nn.init.normal_(self.value.weight, mean=0.0, std=input_std)
        nn.init.normal_(self.output.weight, mean=0.0, std=output_std)
        nn.init.zeros_(self.output.bias)

    @overload
    def forward(
        self,
        x: Tensor,
        cache: AttentionBlockCache | None = None,
        segment_pos: Tensor | None = None,
        *,
        active_mask: Tensor | None = None,
        return_cache: Literal[False] = False,
    ) -> Tensor:
        """Apply local attention without returning an updated cache."""
        ...

    @overload
    def forward(
        self,
        x: Tensor,
        cache: AttentionBlockCache | None = None,
        segment_pos: Tensor | None = None,
        *,
        active_mask: Tensor | None = None,
        return_cache: Literal[True],
    ) -> tuple[Tensor, AttentionBlockCache]:
        """Apply local attention and return its updated KV cache."""
        ...

    def forward(
        self,
        x: Tensor,
        cache: AttentionBlockCache | None = None,
        segment_pos: Tensor | None = None,
        *,
        active_mask: Tensor | None = None,
        return_cache: bool = False,
    ) -> Tensor | tuple[Tensor, AttentionBlockCache]:
        """Apply local MQA and optionally return a bounded cache for later chunks."""
        batch_size, seq_len, _ = x.shape
        if active_mask is None:
            active_mask = torch.ones(batch_size, seq_len, device=x.device, dtype=torch.bool)
        elif active_mask.shape != (batch_size, seq_len):
            raise ValueError("active_mask must match the input [batch, sequence] dimensions")
        else:
            active_mask = active_mask.to(device=x.device, dtype=torch.bool)
        previous_position = None if cache is None else cache.segment_position
        segment_pos = _resolve_segment_positions(
            x, segment_pos, previous_position, active_mask
        )

        queries = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        queries = _apply_rope(queries, segment_pos)
        keys = self.key(x).view(batch_size, seq_len, 1, self.head_dim)
        keys = _apply_rope(keys, segment_pos).squeeze(2)
        values = self.value(x)

        if cache is not None:
            if cache.valid_mask.shape != cache.keys.shape[:2]:
                raise ValueError("Attention cache validity mask has an incompatible shape")
            keys = torch.cat([cache.keys.to(keys), keys], dim=1)
            values = torch.cat([cache.values.to(values), values], dim=1)
            key_valid = torch.cat(
                [cache.valid_mask.to(device=x.device, dtype=torch.bool), active_mask], dim=1
            )
        else:
            key_valid = active_mask

        past_len = keys.size(1) - seq_len
        attended_chunks = []
        for query_start in range(0, seq_len, self.chunk_size):
            query_end = min(seq_len, query_start + self.chunk_size)
            key_end = past_len + query_end
            key_start = max(0, past_len + query_start - self.window_size + 1)
            local_keys = keys[:, key_start:key_end]
            local_values = values[:, key_start:key_end]
            local_key_valid = key_valid[:, key_start:key_end]
            query_indices = past_len + torch.arange(query_start, query_end, device=x.device)
            key_indices = torch.arange(key_start, key_end, device=x.device)
            causal = key_indices[None, :] <= query_indices[:, None]
            query_positions = segment_pos[:, query_start:query_end]
            lookback = torch.clamp(query_positions, max=self.window_size - 1)
            earliest_key = query_indices[None, :] - lookback
            same_document = key_indices[None, None, :] >= earliest_key[..., None]
            query_valid = active_mask[:, query_start:query_end]
            allowed = (
                causal[None, :, :]
                & same_document
                & local_key_valid[:, None, :]
                & query_valid[:, :, None]
            )

            local_queries = queries[:, query_start:query_end]
            scores = torch.einsum("bthd,bsd->bhts", local_queries, local_keys)
            scores = scores * (self.head_dim ** -0.5)
            scores = scores.masked_fill(~allowed[:, None, :, :], float("-inf"))
            weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
            weights = torch.nan_to_num(weights)
            attended_chunks.append(
                torch.einsum("bhts,bsd->bthd", weights, local_values)
            )
        attended = torch.cat(attended_chunks, dim=1)
        output = self.output(attended.reshape(batch_size, seq_len, self.input_dim))
        output = torch.where(active_mask[..., None], output, torch.zeros_like(output))

        if return_cache:
            cache_limit = self.window_size - 1
            valid_counts = key_valid.sum(dim=1).clamp(max=cache_limit)
            cache_length = int(valid_counts.max().item()) if cache_limit else 0
            cached_keys = keys.new_zeros(batch_size, cache_length, self.head_dim)
            cached_values = values.new_zeros(batch_size, cache_length, self.head_dim)
            cached_valid = torch.zeros(
                batch_size, cache_length, device=x.device, dtype=torch.bool
            )
            # Right-align valid entries so physical lookback indices remain aligned
            # across prompts with different lengths in the same batch.
            for batch_index in range(batch_size):
                count = int(valid_counts[batch_index].item())
                if count:
                    selected_keys = keys[batch_index, key_valid[batch_index]][-count:]
                    selected_values = values[batch_index, key_valid[batch_index]][-count:]
                    cached_keys[batch_index, -count:] = selected_keys
                    cached_values[batch_index, -count:] = selected_values
                    cached_valid[batch_index, -count:] = True
            return output, AttentionBlockCache(
                cached_keys,
                cached_values,
                cached_valid,
                _last_active_position(segment_pos, active_mask, previous_position),
            )
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
        final_w_init_variance_scale: float = 1.0,
        scan_mode: ScanMode = "auto",
        attention_chunk_size: int = 128,
    ):
        """Create a pre-normalized recurrent or attention residual block."""
        super().__init__()
        hidden_dim = input_dim * expansion_factor
        self.temporal_block_type = temporal_block_type
        self.norm1 = RMSNorm(input_dim)
        if temporal_block_type == "recurrent":
            self.recurrent = RecurrentBlock(
                input_dim,
                rnn_width,
                gate_blocks=gate_blocks,
                final_w_init_variance_scale=final_w_init_variance_scale,
                scan_mode=scan_mode,
            )
            self.temporal = self.recurrent
        elif temporal_block_type == "attention":
            self.attention = LocalMQAAttention(
                input_dim,
                attention_heads,
                attention_window_size,
                final_w_init_variance_scale,
                attention_chunk_size,
            )
            self.temporal = self.attention
        else:
            raise ValueError(f"Unknown temporal block type: {temporal_block_type}")
        self.norm2 = RMSNorm(input_dim)
        self.mlp = GatedMLPBlock(
            input_dim,
            hidden_dim,
            final_w_init_variance_scale,
        )

    def forward(
        self,
        x: Tensor,
        cache: LayerCache | None = None,
        segment_pos: Tensor | None = None,
        *,
        active_mask: Tensor | None = None,
        return_cache: bool = False,
    ) -> Tensor | tuple[Tensor, LayerCache]:
        """Apply temporal and MLP residual updates, optionally returning layer cache."""
        temporal_result = self.temporal(
            self.norm1(x),
            cache,
            segment_pos,
            active_mask=active_mask,
            return_cache=return_cache,
        )
        if return_cache:
            temporal_update, new_cache = temporal_result
        else:
            temporal_update = temporal_result
            new_cache = None
        x = x + temporal_update
        x = x + self.mlp(self.norm2(x))
        if active_mask is not None:
            x = torch.where(active_mask[..., None], x, torch.zeros_like(x))
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
        scan_mode: ScanMode = "auto",
        attention_chunk_size: int = 128,
    ):
        """Create embeddings, hybrid residual layers, final norm, and tied LM head."""
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive")
        self.vocab_size = vocab_size
        self.input_dim = input_dim
        self.block_types = tuple(block_types) if block_types is not None else _griffin_schedule(depth)
        if len(self.block_types) != depth:
            raise ValueError("block_types must contain exactly depth entries")
        attention_heads = attention_heads or _default_attention_heads(input_dim)
        final_w_init_variance_scale = 2.0 / depth

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
                    final_w_init_variance_scale=final_w_init_variance_scale,
                    scan_mode=scan_mode,
                    attention_chunk_size=attention_chunk_size,
                )
                for block_type in self.block_types
            ]
        )
        self.final_norm = RMSNorm(input_dim)
        self.lm_head = nn.Linear(input_dim, vocab_size, bias=False)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=math.sqrt(1.0 / input_dim))
        if tie_embeddings:
            self.lm_head.weight = self.embd.weight

    @overload
    def forward(
        self,
        token_ids: Tensor,
        cache: Sequence[LayerCache] | None = None,
        segment_pos: Tensor | None = None,
        *,
        token_mask: Tensor | None = None,
        return_cache: Literal[False] = False,
    ) -> Tensor:
        """Return logits without constructing updated layer caches."""
        ...

    @overload
    def forward(
        self,
        token_ids: Tensor,
        cache: Sequence[LayerCache] | None = None,
        segment_pos: Tensor | None = None,
        *,
        token_mask: Tensor | None = None,
        return_cache: Literal[True],
    ) -> tuple[Tensor, list[LayerCache]]:
        """Return logits and one updated cache per residual layer."""
        ...

    def forward(
        self,
        token_ids: Tensor,
        cache: Sequence[LayerCache] | None = None,
        segment_pos: Tensor | None = None,
        *,
        token_mask: Tensor | None = None,
        return_cache: bool = False,
    ) -> Tensor | tuple[Tensor, list[LayerCache]]:
        """Embed token IDs and return logits, optionally with bounded decode caches."""
        if cache is not None and len(cache) != len(self.layers):
            raise ValueError("cache must contain one entry per residual layer")
        if segment_pos is not None and segment_pos.shape != token_ids.shape:
            raise ValueError("segment_pos must have the same [batch, sequence] shape as token_ids")
        if token_mask is not None and token_mask.shape != token_ids.shape:
            raise ValueError("token_mask must have the same shape as token_ids")
        if token_mask is not None:
            token_mask = token_mask.to(device=token_ids.device, dtype=torch.bool)
        if (
            token_mask is not None
            and token_mask.size(1) > 1
            and bool((~token_mask[:, :-1] & token_mask[:, 1:]).any())
        ):
            raise ValueError("token_mask must describe right-padded sequences")

        # Scaling restores unit-variance input activations after LeCun embedding init.
        x = self.embd(token_ids) * self.embedding_scale
        if token_mask is not None:
            x = torch.where(token_mask[..., None], x, torch.zeros_like(x))
        new_cache: list[LayerCache] = []
        for index, layer in enumerate(self.layers):
            layer_cache = None if cache is None else cache[index]
            if return_cache:
                x, updated_cache = layer(
                    x,
                    layer_cache,
                    segment_pos,
                    active_mask=token_mask,
                    return_cache=True,
                )
                new_cache.append(updated_cache)
            else:
                x = layer(
                    x,
                    layer_cache,
                    segment_pos,
                    active_mask=token_mask,
                )
        logits = self.lm_head(self.final_norm(x))
        if return_cache:
            return logits, new_cache
        return logits
