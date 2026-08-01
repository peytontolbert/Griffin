import torch

from griffin import RecurrentBlock


def test_recurrent_block_preserves_input_shape():
    block = RecurrentBlock(input_dim=10, rnn_width=13)
    x = torch.randn(2, 7, 10)

    output = block(x)

    assert output.shape == x.shape


def test_future_token_does_not_change_past_outputs():
    torch.manual_seed(0)
    block = RecurrentBlock(input_dim=6, rnn_width=6).eval()
    x = torch.randn(2, 8, 6)
    changed = x.clone()
    changed[:, 4, :] += 10.0

    with torch.no_grad():
        original_output = block(x)
        changed_output = block(changed)

    torch.testing.assert_close(original_output[:, :4, :], changed_output[:, :4, :])
