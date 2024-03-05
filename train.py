import torch
import random
from transformers import BertTokenizer, BertForQuestionAnswering
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from griffin.griffin import GriffinModel  # Replace with your model import
import gzip
import numpy as np
device = torch.device("cuda:0")
# Hyperparameters
learning_rate = 1e-3
batch_size = 4
num_epochs = 10
input_dim = 768
rnn_width = 1024
depth = 12
mlp_expansion_factor = 3
VALIDATE_EVERY = 100
GENERATE_EVERY = 500
GENERATE_LENGTH = 1024
SEQ_LEN = 256
# Initialize the BERT tokenizer
# Load Wikipedia dataset from `datasets`

# Load the SQuAD dataset
dataset = load_dataset("imdb", split='train')


# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# Tokenize and prepare dataset
def encode(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        padding="max_length",
        max_length=SEQ_LEN,
        return_tensors="pt",
    )


encoded_dataset = dataset.map(encode, batched=True)
encoded_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "label"]
)

# Split dataset into training and validation
train_size = int(0.9 * len(encoded_dataset))
val_size = len(encoded_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    encoded_dataset, [train_size, val_size]
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
vocab_size = tokenizer.vocab_size
# Model, optimizer, and loss function
model = GriffinModel(vocab_size, input_dim, mlp_expansion_factor, rnn_width, depth).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()  # Replace with your loss function
# Training loop
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(train_loader):
        # Forward pass
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)  # Use this as targets in your loss calculation
        optimizer.zero_grad()
        x = input_ids
        outputs = model(x)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        if batch_idx % VALIDATE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                loss = model(next(val_loader))
                print(f"validation loss: {loss.mean().item()}")

        # Print loss every 100 batches
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Batch {batch_idx}, Loss: {loss.item()}"
            )
        # Save the model
        torch.save(model.state_dict(), "your_model.pth")

# Save the model
torch.save(model.state_dict(), "your_model.pth")
