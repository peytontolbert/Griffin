"""Dataset helpers for next-token language-model training."""

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Slice a flat token stream into shifted input and target windows."""

    def __init__(self, encoded_text, block_size: int):
        """Store encoded token IDs and the fixed training sequence length."""
        self.data = torch.as_tensor(encoded_text, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        """Return the number of shifted windows available in the token stream."""
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx: int):
        """Return ``x`` and next-token targets ``y`` for one contiguous window."""
        chunk = self.data[idx : idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]
