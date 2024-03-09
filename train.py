import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from dataset import TextDataset
from griffin.griffin import GriffinModel  # Replace with your model import
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
def map_pair(example):
    return {'question': example['question'], 'answer': example['answer']}
# Load the SQuAD dataset
dataset = load_dataset("wiki_qa", split='train')
for data in dataset:
    print(data)
    pairs = data.map(map_pair, batched=False)
print(f'pairs: {pairs}')
pairlist = []
for pair in pairs:
    qa = pair['question']+pair['answer']
    pairlist.append(qa)
chars = sorted(list(set(pairlist)))
vocab_size = len(pairlist)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Let's now split up the data into train and validation sets
answers = torch.tensor(encode(qa['question']), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# Create datasets
train_dataset = TextDataset(train_data, SEQ_LEN)
val_dataset = TextDataset(val_data, SEQ_LEN)

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


# Model, optimizer, and loss function
model = GriffinModel(vocab_size, input_dim, mlp_expansion_factor, rnn_width, depth)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()  # Replace with your loss function
# Training loop
for epoch in range(num_epochs):
    for batch, label in train_dataloader:
        # Forward pass
        batch, label = batch.to(device), label.to(device)
        labels = batch['answer'].to(device)  # Use this as targets in your loss calculation
        optimizer.zero_grad()
        outputs = model(batch)
        print(outputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        if batch % VALIDATE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                loss = model(next(val_dataloader))
                print(f"validation loss: {loss.mean().item()}")

        # Print loss every 100 batches
        if batch % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Batch {batch}, Loss: {loss.item()}"
            )
        # Save the model
        torch.save(model.state_dict(), "your_model.pth")

# Save the model
torch.save(model.state_dict(), "your_model.pth")
