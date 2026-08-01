"""Dataset helpers for next-token language-model training."""

from itertools import islice
from typing import Any, Iterable, Mapping

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info


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


class PackedTokenDataset(IterableDataset):
    """Tokenize and pack a text-row stream into non-overlapping LM sequences."""

    def __init__(
        self,
        rows: Iterable[Mapping[str, Any]],
        tokenizer: Any,
        block_size: int,
        *,
        text_column: str = "text",
        max_tokens: int | None = None,
    ):
        """Store a restartable row source and tokenizer-backed packing settings."""
        super().__init__()
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.rows = rows
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.text_column = text_column
        self.max_tokens = max_tokens

    def _rows_for_worker(self) -> Iterable[Mapping[str, Any]]:
        """Shard rows across DataLoader workers when worker processes are enabled."""
        worker = get_worker_info()
        if worker is None:
            return iter(self.rows)
        if hasattr(self.rows, "shard"):
            return iter(self.rows.shard(num_shards=worker.num_workers, index=worker.id))
        return islice(iter(self.rows), worker.id, None, worker.num_workers)

    def __iter__(self):
        """Yield shifted token blocks without allowing train/validation overlap."""
        buffer: list[int] = []
        token_count = 0
        eos_token_id = self.tokenizer.eos_token_id
        for row in self._rows_for_worker():
            text = row.get(self.text_column)
            if not isinstance(text, str) or not text:
                continue
            token_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            if eos_token_id is not None:
                token_ids.append(eos_token_id)
            if self.max_tokens is not None:
                remaining = self.max_tokens - token_count
                if remaining <= 0:
                    break
                token_ids = token_ids[:remaining]
            buffer.extend(token_ids)
            token_count += len(token_ids)

            while len(buffer) >= self.block_size + 1:
                chunk = torch.tensor(buffer[: self.block_size + 1], dtype=torch.long)
                yield chunk[:-1], chunk[1:]
                # Retain the last target as the first input of the next block.
                del buffer[: self.block_size]
