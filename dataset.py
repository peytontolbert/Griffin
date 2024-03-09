from torch.utils.data import Dataset
class TextDataset(Dataset):
    def __init__(self, encoded_text, block_size):
        # Assuming encoded_text is a list of integers representing encoded characters
        self.data = encoded_text
        self.block_size = block_size

    def __len__(self):
        # The length is the number of blocks we can make
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        # Get the sequence of tokens that starts at this index
        chunk = self.data[idx:idx + self.block_size + 1]
        # Input sequence (x) is the first block_size characters
        # Target sequence (y) is the last block_size characters
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

