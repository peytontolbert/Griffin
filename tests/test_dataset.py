"""Tests for packed token IDs and per-document segment positions."""

import torch

from dataset import PackedTokenDataset


class _CharacterTokenizer:
    """Minimal deterministic tokenizer used by packed-dataset tests."""

    eos_token_id = 99

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        """Map the test alphabet to compact integer token IDs."""
        assert not add_special_tokens
        return {"input_ids": [ord(character) - ord("a") + 1 for character in text]}


def test_packed_dataset_resets_positions_for_each_row():
    """A new source row must restart segment positions at zero inside a block."""
    rows = [{"text": "ab"}, {"text": "cde"}]
    dataset = PackedTokenDataset(rows, _CharacterTokenizer(), block_size=5)

    x, y, segment_pos = next(iter(dataset))

    torch.testing.assert_close(x, torch.tensor([1, 2, 99, 3, 4]))
    torch.testing.assert_close(y, torch.tensor([2, 99, 3, 4, 5]))
    torch.testing.assert_close(segment_pos, torch.tensor([0, 1, 2, 0, 1]))


def test_packed_dataset_resets_each_training_block():
    """A block beginning mid-document has no cache and must start at position zero."""
    rows = [{"text": "abcdefghij"}]
    dataset = PackedTokenDataset(rows, _CharacterTokenizer(), block_size=4)
    iterator = iter(dataset)

    _, _, first_positions = next(iterator)
    _, _, second_positions = next(iterator)

    torch.testing.assert_close(first_positions, torch.tensor([0, 1, 2, 3]))
    torch.testing.assert_close(second_positions, torch.tensor([0, 1, 2, 3]))
