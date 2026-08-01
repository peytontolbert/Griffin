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


def test_segment_reset_blocks_previous_document_context():
    """Logits after a reset must be independent of every previous-document token."""
    torch.manual_seed(0)
    model = GriffinModel(
        vocab_size=23,
        input_dim=8,
        mlp_expansion_factor=2,
        rnn_width=12,
        depth=3,
        attention_heads=2,
        attention_window_size=4,
    ).eval()
    suffix = torch.tensor([[5, 6, 7]])
    first = torch.cat([torch.tensor([[1, 2, 3, 4]]), suffix], dim=1)
    changed = torch.cat([torch.tensor([[9, 10, 11, 4]]), suffix], dim=1)
    segment_pos = torch.tensor([[0, 1, 2, 3, 0, 1, 2]])

    with torch.no_grad():
        first_logits = model(first, segment_pos=segment_pos)
        changed_logits = model(changed, segment_pos=segment_pos)

    torch.testing.assert_close(first_logits[:, 4:], changed_logits[:, 4:])


def test_cached_segment_reset_matches_full_forward():
    """A reset inside a later cached chunk must match full-sequence execution."""
    torch.manual_seed(0)
    model = GriffinModel(
        vocab_size=23,
        input_dim=8,
        mlp_expansion_factor=2,
        rnn_width=12,
        depth=3,
        attention_heads=2,
        attention_window_size=4,
    ).eval()
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    segment_pos = torch.tensor([[0, 1, 2, 3, 0, 1, 2]])

    with torch.no_grad():
        full_logits = model(tokens, segment_pos=segment_pos)
        first, cache = model(
            tokens[:, :3],
            segment_pos=segment_pos[:, :3],
            return_cache=True,
        )
        second, _ = model(
            tokens[:, 3:],
            cache,
            segment_pos[:, 3:],
            return_cache=True,
        )

    torch.testing.assert_close(
        full_logits,
        torch.cat([first, second], dim=1),
        atol=1e-6,
        rtol=1e-6,
    )


def test_variable_length_batched_prefill_and_decode_match_individual_sequences():
    """Right-padded prompts must retain independent recurrent and attention caches."""
    torch.manual_seed(0)
    model = GriffinModel(
        vocab_size=23,
        input_dim=8,
        mlp_expansion_factor=2,
        rnn_width=12,
        depth=3,
        attention_heads=2,
        attention_window_size=4,
        attention_chunk_size=2,
    ).eval()
    prompts = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 0, 0]])
    prompt_mask = torch.tensor(
        [[True, True, True, True, True], [True, True, True, False, False]]
    )
    next_tokens = torch.tensor([[9], [10]])

    with torch.no_grad():
        batched_logits, batched_cache = model(
            prompts,
            token_mask=prompt_mask,
            return_cache=True,
        )
        batched_next, _ = model(next_tokens, batched_cache, return_cache=True)

        first_logits, first_cache = model(prompts[:1], return_cache=True)
        first_next, _ = model(next_tokens[:1], first_cache, return_cache=True)
        second_logits, second_cache = model(prompts[1:2, :3], return_cache=True)
        second_next, _ = model(next_tokens[1:2], second_cache, return_cache=True)

    torch.testing.assert_close(batched_logits[:1], first_logits, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        batched_logits[1:2, :3], second_logits, atol=2e-6, rtol=2e-6
    )
    torch.testing.assert_close(batched_next[:1], first_next, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(batched_next[1:2], second_next, atol=2e-6, rtol=2e-6)
