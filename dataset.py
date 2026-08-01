import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, encoded_text, block_size: int):
        self.data = torch.as_tensor(encoded_text, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx: int):
        chunk = self.data[idx : idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]
