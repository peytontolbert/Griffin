"""Behavioral tests for the top-level Griffin language model."""

import torch
import torch.nn.functional as F

from griffin import GriffinModel


def test_model_returns_vocab_logits():
    """The model should return raw vocabulary logits, not hidden-width probs."""
    model = GriffinModel(
        vocab_size=17,
        input_dim=8,
        mlp_expansion_factor=2,
        rnn_width=8,
        depth=2,
    )
    tokens = torch.randint(0, 17, (2, 5))

    logits = model(tokens)

    assert logits.shape == (2, 5, 17)
    assert not torch.allclose(logits.sum(dim=-1), torch.ones(2, 5))


def test_eval_forward_is_deterministic():
    """Registered heads should make eval output deterministic for same input."""
    torch.manual_seed(0)
    model = GriffinModel(
        vocab_size=17,
        input_dim=8,
        mlp_expansion_factor=2,
        rnn_width=8,
        depth=1,
    ).eval()
    tokens = torch.randint(0, 17, (2, 5))

    with torch.no_grad():
        first = model(tokens)
        second = model(tokens)

    torch.testing.assert_close(first, second)


def test_lm_head_is_registered_and_receives_gradients():
    """The language-model head must be trainable by the optimizer."""
    model = GriffinModel(
        vocab_size=17,
        input_dim=8,
        mlp_expansion_factor=2,
        rnn_width=8,
        depth=1,
    )
    tokens = torch.randint(0, 17, (2, 5))
    targets = torch.randint(0, 17, (2, 5))

    logits = model(tokens)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    loss.backward()

    assert "lm_head.weight" in dict(model.named_parameters())
    assert model.lm_head.weight.grad is not None
    assert model.lm_head.weight.grad.abs().sum() > 0
