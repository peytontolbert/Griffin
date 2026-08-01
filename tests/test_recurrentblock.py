"""Behavioral tests for the causal recurrent block."""

import torch

from griffin import RecurrentBlock


def test_recurrent_block_preserves_input_shape():
    """The recurrent block should preserve batch, sequence, and model width."""
    block = RecurrentBlock(input_dim=10, rnn_width=13)
    x = torch.randn(2, 7, 10)

    output = block(x)

    assert output.shape == x.shape


def test_future_token_does_not_change_past_outputs():
    """Changing a future token must not alter earlier recurrent-block outputs."""
    torch.manual_seed(0)
    block = RecurrentBlock(input_dim=6, rnn_width=6).eval()
    x = torch.randn(2, 8, 6)
    changed = x.clone()
    changed[:, 4, :] += 10.0

    with torch.no_grad():
        original_output = block(x)
        changed_output = block(changed)

    torch.testing.assert_close(original_output[:, :4, :], changed_output[:, :4, :])


def test_cached_recurrent_chunks_match_full_sequence():
    """The cache must preserve both convolution history and RG-LRU state."""
    torch.manual_seed(0)
    block = RecurrentBlock(input_dim=8, rnn_width=12).eval()
    x = torch.randn(2, 9, 8)

    with torch.no_grad():
        full_output = block(x)
        first, cache = block(x[:, :3], return_cache=True)
        second, cache = block(x[:, 3:7], cache, return_cache=True)
        third, _ = block(x[:, 7:], cache, return_cache=True)

    torch.testing.assert_close(
        full_output,
        torch.cat([first, second, third], dim=1),
        atol=1e-6,
        rtol=1e-6,
    )
