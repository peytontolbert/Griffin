"""Behavioral tests for the real-gated linear recurrent unit."""

import copy

import torch

from griffin import BlockDiagonalLinear, RG_LRU


def test_rglru_earlier_token_changes_later_hidden_state():
    """Earlier tokens should influence later states through the recurrent scan."""
    torch.manual_seed(0)
    module = RG_LRU(rnn_width=6).eval()
    x = torch.randn(2, 5, 6)
    changed = x.clone()
    changed[:, 0, :] += 3.0

    with torch.no_grad():
        original_output = module(x)
        changed_output = module(changed)

    later_diff = (original_output[:, 1:, :] - changed_output[:, 1:, :]).abs().max()
    assert later_diff > 1e-6


def test_rglru_returns_final_state():
    """The returned final state should match the last sequence output."""
    torch.manual_seed(0)
    module = RG_LRU(rnn_width=6).eval()
    x = torch.randn(2, 5, 6)

    with torch.no_grad():
        output, final_state = module(x, return_state=True)

    assert output.shape == x.shape
    assert final_state.shape == (2, 6)
    torch.testing.assert_close(output[:, -1, :], final_state)


def test_rglru_uses_block_diagonal_lecun_gates():
    """Paper-scale widths should use 16 independently initialized gate blocks."""
    torch.manual_seed(0)
    module = RG_LRU(rnn_width=64, gate_blocks=16)

    assert isinstance(module.recurrence_gate, BlockDiagonalLinear)
    assert isinstance(module.input_gate, BlockDiagonalLinear)
    assert module.recurrence_gate.weight.shape == (16, 4, 4)
    expected_std = 0.5
    assert abs(module.recurrence_gate.weight.std().item() - expected_std) < 0.08


def test_rglru_bounds_bfloat16_sqrt_gradients():
    """Near-unit recurrence decay must not create NaN gradients in bfloat16."""
    module = RG_LRU(rnn_width=16, gate_blocks=4).to(torch.bfloat16)
    with torch.no_grad():
        module.recurrence_gate.weight.zero_()
        module.recurrence_gate.bias.fill_(-100)
        module.Lambda.fill_(20)
    x = torch.randn(2, 8, 16, dtype=torch.bfloat16, requires_grad=True)

    output = module(x)
    output.float().square().mean().backward()

    assert torch.isfinite(output).all()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(module.Lambda.grad).all()


def test_associative_scan_matches_sequential_values_and_gradients():
    """Parallel affine composition must preserve recurrent outputs and gradients."""
    torch.manual_seed(0)
    sequential = RG_LRU(rnn_width=8, gate_blocks=4, scan_mode="sequential")
    associative = copy.deepcopy(sequential)
    associative.scan_mode = "associative"
    sequential_input = torch.randn(2, 11, 8, requires_grad=True)
    associative_input = sequential_input.detach().clone().requires_grad_(True)
    reset_mask = torch.zeros(2, 11, dtype=torch.bool)
    reset_mask[:, 6] = True

    sequential_output = sequential(sequential_input, reset_mask=reset_mask)
    associative_output = associative(associative_input, reset_mask=reset_mask)
    sequential_output.square().mean().backward()
    associative_output.square().mean().backward()

    torch.testing.assert_close(sequential_output, associative_output, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        sequential_input.grad,
        associative_input.grad,
        atol=2e-6,
        rtol=2e-6,
    )
    for sequential_parameter, associative_parameter in zip(
        sequential.parameters(), associative.parameters()
    ):
        torch.testing.assert_close(
            sequential_parameter.grad,
            associative_parameter.grad,
            atol=2e-6,
            rtol=2e-6,
        )
