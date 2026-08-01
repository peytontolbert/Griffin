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


def test_lm_head_is_tied_and_receives_gradients():
    """The output projection should share the trainable embedding parameter."""
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

    assert model.lm_head.weight is model.embd.weight
    assert model.embd.weight.grad is not None
    assert model.embd.weight.grad.abs().sum() > 0


def test_default_model_uses_griffin_hybrid_schedule():
    """Every third temporal layer should be local attention by default."""
    model = GriffinModel(
        vocab_size=17,
        input_dim=8,
        mlp_expansion_factor=2,
        rnn_width=8,
        depth=6,
        attention_heads=2,
    )

    assert model.block_types == (
        "recurrent",
        "recurrent",
        "attention",
        "recurrent",
        "recurrent",
        "attention",
    )


def test_cached_token_decoding_matches_full_hybrid_forward():
    """A complete Griffin cache should preserve exact autoregressive semantics."""
    torch.manual_seed(0)
    model = GriffinModel(
        vocab_size=17,
        input_dim=8,
        mlp_expansion_factor=2,
        rnn_width=12,
        depth=3,
        attention_heads=2,
        attention_window_size=4,
    ).eval()
    tokens = torch.randint(0, 17, (2, 7))

    with torch.no_grad():
        full_logits = model(tokens)
        cache = None
        token_logits = []
        for position in range(tokens.size(1)):
            logits, cache = model(
                tokens[:, position : position + 1],
                cache,
                return_cache=True,
            )
            token_logits.append(logits)

    torch.testing.assert_close(
        full_logits,
        torch.cat(token_logits, dim=1),
        atol=2e-6,
        rtol=2e-6,
    )
