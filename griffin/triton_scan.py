"""Optional fused Triton kernels for the diagonal RG-LRU recurrence.

The kernels keep one feature tile of recurrent state in registers while they
walk the sequence. This performs linear work in sequence length and launches
one forward and one backward kernel instead of one operation per token.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

try:
    import triton
    import triton.language as tl
except ImportError:  # Triton is optional so CPU installations remain usable.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _rglru_scan_forward_kernel(
        decay_pointer,
        input_pointer,
        initial_state_pointer,
        output_pointer,
        sequence_length,
        width,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Scan one batch/feature tile forward while retaining state in registers."""
        batch_index = tl.program_id(0)
        feature_offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        feature_mask = feature_offsets < width
        state_offsets = batch_index * width + feature_offsets
        state = tl.load(initial_state_pointer + state_offsets, mask=feature_mask, other=0.0)
        state = state.to(tl.float32)

        position = 0
        while position < sequence_length:
            offsets = (
                (batch_index * sequence_length + position) * width + feature_offsets
            )
            decay = tl.load(decay_pointer + offsets, mask=feature_mask, other=0.0)
            current_input = tl.load(
                input_pointer + offsets, mask=feature_mask, other=0.0
            )
            state = decay.to(tl.float32) * state + current_input.to(tl.float32)
            tl.store(output_pointer + offsets, state, mask=feature_mask)
            position += 1


    @triton.jit
    def _rglru_scan_backward_kernel(
        decay_pointer,
        initial_state_pointer,
        output_pointer,
        output_gradient_pointer,
        decay_gradient_pointer,
        input_gradient_pointer,
        initial_state_gradient_pointer,
        sequence_length,
        width,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Apply reverse-mode differentiation to one batch/feature recurrence tile."""
        batch_index = tl.program_id(0)
        feature_offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        feature_mask = feature_offsets < width
        state_offsets = batch_index * width + feature_offsets
        initial_state = tl.load(
            initial_state_pointer + state_offsets, mask=feature_mask, other=0.0
        ).to(tl.float32)
        state_gradient = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        position = sequence_length
        while position > 0:
            position -= 1
            offsets = (
                (batch_index * sequence_length + position) * width + feature_offsets
            )
            output_gradient = tl.load(
                output_gradient_pointer + offsets, mask=feature_mask, other=0.0
            )
            state_gradient += output_gradient.to(tl.float32)

            previous_offsets = offsets - width
            previous_state = tl.load(
                output_pointer + previous_offsets,
                mask=feature_mask & (position > 0),
                other=0.0,
            ).to(tl.float32)
            previous_state = tl.where(position > 0, previous_state, initial_state)
            tl.store(
                decay_gradient_pointer + offsets,
                state_gradient * previous_state,
                mask=feature_mask,
            )
            tl.store(
                input_gradient_pointer + offsets,
                state_gradient,
                mask=feature_mask,
            )

            decay = tl.load(decay_pointer + offsets, mask=feature_mask, other=0.0)
            state_gradient *= decay.to(tl.float32)

        tl.store(
            initial_state_gradient_pointer + state_offsets,
            state_gradient,
            mask=feature_mask,
        )


def triton_scan_available() -> bool:
    """Return whether the optional Triton package can provide the fused kernels."""
    return triton is not None


class _FusedRGLRUScan(torch.autograd.Function):
    """Autograd bridge for the fused Triton recurrence kernels."""

    @staticmethod
    def forward(ctx, decay: Tensor, current_input: Tensor, initial_state: Tensor) -> Tensor:
        """Launch the fused forward scan and retain tensors needed by backward."""
        if triton is None:
            raise RuntimeError("Fused RG-LRU scan requires the optional Triton package")
        if not decay.is_cuda or not current_input.is_cuda or not initial_state.is_cuda:
            raise RuntimeError("Fused RG-LRU scan requires CUDA tensors")
        if decay.shape != current_input.shape or decay.dim() != 3:
            raise ValueError("decay and current_input must share shape [batch, sequence, width]")
        if initial_state.shape != (decay.size(0), decay.size(2)):
            raise ValueError("initial_state must have shape [batch, width]")
        if decay.dtype != torch.float32 or current_input.dtype != torch.float32:
            raise TypeError("Fused RG-LRU scan expects fp32 decay and normalized input")
        if initial_state.dtype != torch.float32:
            raise TypeError("Fused RG-LRU scan expects fp32 recurrent state")

        decay = decay.contiguous()
        current_input = current_input.contiguous()
        initial_state = initial_state.contiguous()
        output = torch.empty_like(current_input, dtype=torch.float32)
        batch_size, sequence_length, width = decay.shape
        block_size = min(256, triton.next_power_of_2(width))
        grid = (batch_size, triton.cdiv(width, block_size))
        _rglru_scan_forward_kernel[grid](
            decay,
            current_input,
            initial_state,
            output,
            sequence_length,
            width,
            BLOCK_SIZE=block_size,
            num_warps=4,
        )
        ctx.save_for_backward(decay, initial_state, output)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, output_gradient: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Launch the fused first-order reverse recurrence for differentiable inputs."""
        decay, initial_state, output = ctx.saved_tensors
        output_gradient = output_gradient.contiguous()
        decay_gradient = torch.empty_like(decay)
        input_gradient = torch.empty_like(output)
        initial_state_gradient = torch.empty_like(initial_state)
        batch_size, sequence_length, width = decay.shape
        block_size = min(256, triton.next_power_of_2(width))
        grid = (batch_size, triton.cdiv(width, block_size))
        _rglru_scan_backward_kernel[grid](
            decay,
            initial_state,
            output,
            output_gradient,
            decay_gradient,
            input_gradient,
            initial_state_gradient,
            sequence_length,
            width,
            BLOCK_SIZE=block_size,
            num_warps=4,
        )
        return decay_gradient, input_gradient, initial_state_gradient


def fused_rglru_scan(decay: Tensor, current_input: Tensor, initial_state: Tensor) -> Tensor:
    """Run the differentiable fused CUDA scan for a diagonal affine recurrence."""
    return _FusedRGLRUScan.apply(decay, current_input, initial_state)
