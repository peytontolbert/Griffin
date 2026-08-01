"""Behavioral tests for the real-gated linear recurrent unit."""

import torch

from griffin import RG_LRU


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
