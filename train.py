from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader

from dataset import TextDataset
from griffin import GriffinModel


def build_corpus(split: str) -> str:
    dataset = load_dataset("wiki_qa", split=split)
    examples = []
    for row in dataset:
        question = row.get("question")
        answer = row.get("answer")
        if question and answer:
            examples.append(f"Question: {question}\nAnswer: {answer}\n\n")
    if not examples:
        raise RuntimeError(f"No usable question/answer rows found in split {split!r}")
    return "".join(examples)


def build_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: dict[str, int]) -> torch.Tensor:
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)


def cross_entropy_for_sequence_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


@torch.no_grad()
def estimate_loss(
    model: GriffinModel,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> float:
    model.eval()
    losses = []
    for batch_idx, (x, y) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        losses.append(cross_entropy_for_sequence_logits(logits, y).item())
    model.train()
    return sum(losses) / max(1, len(losses))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small Griffin-style character LM.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--input-dim", type=int, default=768)
    parser.add_argument("--rnn-width", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--mlp-expansion-factor", type=int, default=3)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--validation-batches", type=int, default=10)
    parser.add_argument("--checkpoint", type=Path, default=Path("griffin_model.pth"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    corpus = build_corpus(args.split)
    stoi, _ = build_vocab(corpus)
    token_ids = encode(corpus, stoi)
    if len(token_ids) <= args.seq_len + 1:
        raise RuntimeError("Corpus is too short for the configured sequence length")

    split_idx = int(0.9 * len(token_ids))
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[max(0, split_idx - args.seq_len) :]

    train_dataset = TextDataset(train_ids, args.seq_len)
    val_dataset = TextDataset(val_ids, args.seq_len)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError("Dataset split is too small for the configured sequence length")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = GriffinModel(
        vocab_size=len(stoi),
        input_dim=args.input_dim,
        mlp_expansion_factor=args.mlp_expansion_factor,
        rnn_width=args.rnn_width,
        depth=args.depth,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    step = 0
    model.train()
    for epoch in range(args.epochs):
        for x, y in train_dataloader:
            step += 1
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = cross_entropy_for_sequence_logits(logits, y)
            loss.backward()
            optimizer.step()

            if step % args.validate_every == 0:
                val_loss = estimate_loss(
                    model,
                    val_dataloader,
                    device,
                    args.validation_batches,
                )
                print(
                    f"epoch={epoch + 1} step={step} "
                    f"train_loss={loss.item():.4f} val_loss={val_loss:.4f}"
                )

    torch.save(
        {
            "model": model.state_dict(),
            "stoi": stoi,
            "config": vars(args),
        },
        args.checkpoint,
    )


if __name__ == "__main__":
    main()
