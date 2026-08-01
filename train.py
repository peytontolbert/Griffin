"""Train Griffin on packed token IDs with paper-oriented model defaults.

The paper's MassiveText corpus is not public, so exact data reproduction is not
possible from this repository. The trainer accepts any Hugging Face text dataset
and tokenizer, uses the paper's 100M-scale dimensions and 2048-token context by
default, and includes an offline smoke mode for end-to-end verification.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import DatasetDict, IterableDatasetDict, load_dataset
from torch.utils.data import DataLoader

from dataset import PackedTokenDataset, TextDataset
from griffin import GriffinModel


def cross_entropy_for_sequence_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute next-token CE for ``[batch, sequence, vocab]`` logits."""
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def unpack_language_model_batch(
    batch: list[torch.Tensor] | tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return inputs, targets, and segment positions from packed or flat data."""
    if len(batch) == 3:
        x, y, segment_pos = batch
        return x, y, segment_pos
    if len(batch) == 2:
        x, y = batch
        positions = torch.arange(x.size(1), dtype=torch.long)
        return x, y, positions[None, :].expand(x.size(0), -1)
    raise ValueError("Language-model batches must contain two or three tensors")


@torch.no_grad()
def estimate_loss(
    model: GriffinModel,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> float:
    """Estimate validation loss and restore the model's previous train/eval mode."""
    was_training = model.training
    model.eval()
    losses = []
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        x, y, segment_pos = unpack_language_model_batch(batch)
        x = x.to(device)
        y = y.to(device)
        segment_pos = segment_pos.to(device)
        logits = model(x, segment_pos=segment_pos)
        losses.append(cross_entropy_for_sequence_logits(logits, y).item())
    model.train(was_training)
    if not losses:
        raise RuntimeError("Validation dataset produced no complete token blocks")
    return sum(losses) / len(losses)


def parse_args() -> argparse.Namespace:
    """Parse model, data, optimization, and smoke-test settings."""
    parser = argparse.ArgumentParser(description="Train the hybrid Griffin language model.")
    parser.add_argument("--dataset-name", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--tokenizer-name", default="gpt2")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-tokens", type=int)
    parser.add_argument("--max-validation-tokens", type=int, default=1_000_000)

    # Table 2's 100M model uses D=768, D_RNN=1024, N=12, and M=3.
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--input-dim", type=int, default=768)
    parser.add_argument("--rnn-width", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--mlp-expansion-factor", type=int, default=3)
    parser.add_argument("--attention-heads", type=int, default=6)
    parser.add_argument("--attention-window-size", type=int, default=1024)
    parser.add_argument("--gate-blocks", type=int, default=16)

    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--validation-batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=Path("griffin_model.pth"))
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def _load_real_dataloaders(
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader, int, Any]:
    """Load separate text splits and create streaming packed-token loaders."""
    from transformers import AutoTokenizer

    dataset_config = args.dataset_config or None
    splits = load_dataset(
        args.dataset_name,
        dataset_config,
        streaming=args.streaming,
    )
    if not isinstance(splits, (DatasetDict, IterableDatasetDict)):
        raise RuntimeError("Expected load_dataset without split= to return named splits")
    if args.train_split not in splits:
        raise ValueError(f"Dataset has no train split named {args.train_split!r}")
    if args.validation_split not in splits:
        if args.streaming:
            raise ValueError(
                "A separate validation split is required in streaming mode; "
                "disable streaming to derive one from train"
            )
        divided = splits[args.train_split].train_test_split(test_size=0.01, seed=args.seed)
        train_rows, validation_rows = divided["train"], divided["test"]
    else:
        train_rows = splits[args.train_split]
        validation_rows = splits[args.validation_split]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("The selected tokenizer must define an EOS token")
    train_dataset = PackedTokenDataset(
        train_rows,
        tokenizer,
        args.seq_len,
        text_column=args.text_column,
        max_tokens=args.max_train_tokens,
    )
    validation_dataset = PackedTokenDataset(
        validation_rows,
        tokenizer,
        args.seq_len,
        text_column=args.text_column,
        max_tokens=args.max_validation_tokens,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size)
    return train_loader, validation_loader, len(tokenizer), tokenizer


def _load_smoke_dataloaders(
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader, int, None]:
    """Create deterministic, disjoint synthetic streams without network access."""
    generator = torch.Generator().manual_seed(args.seed)
    vocab_size = 128
    sequence_length = 16
    train_ids = torch.randint(0, vocab_size, (512,), generator=generator)
    validation_ids = torch.randint(0, vocab_size, (256,), generator=generator)
    train_loader = DataLoader(
        TextDataset(train_ids, sequence_length),
        batch_size=2,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        TextDataset(validation_ids, sequence_length),
        batch_size=2,
    )
    return train_loader, validation_loader, vocab_size, None


def _model_from_args(args: argparse.Namespace, vocab_size: int) -> GriffinModel:
    """Construct either the requested paper-scale model or a tiny hybrid smoke model."""
    if args.smoke_test:
        return GriffinModel(
            vocab_size=vocab_size,
            input_dim=32,
            mlp_expansion_factor=2,
            rnn_width=48,
            depth=3,
            attention_heads=4,
            attention_window_size=8,
            gate_blocks=8,
        )
    return GriffinModel(
        vocab_size=vocab_size,
        input_dim=args.input_dim,
        mlp_expansion_factor=args.mlp_expansion_factor,
        rnn_width=args.rnn_width,
        depth=args.depth,
        attention_heads=args.attention_heads,
        attention_window_size=args.attention_window_size,
        gate_blocks=args.gate_blocks,
    )


def train(args: argparse.Namespace) -> tuple[GriffinModel, float]:
    """Run bounded next-token training and return the model and final loss."""
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.smoke_test:
        train_loader, validation_loader, vocab_size, tokenizer = _load_smoke_dataloaders(args)
        max_steps = min(args.max_steps, 2)
        validate_every = 1
        validation_batches = 1
    else:
        train_loader, validation_loader, vocab_size, tokenizer = _load_real_dataloaders(args)
        max_steps = args.max_steps
        validate_every = args.validate_every
        validation_batches = args.validation_batches

    model = _model_from_args(args, vocab_size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    model.train()
    final_loss = float("nan")
    for step, batch in enumerate(train_loader, start=1):
        if step > max_steps:
            break
        x, y, segment_pos = unpack_language_model_batch(batch)
        x = x.to(device)
        y = y.to(device)
        segment_pos = segment_pos.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, segment_pos=segment_pos)
        loss = cross_entropy_for_sequence_logits(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        final_loss = loss.item()

        if step % validate_every == 0:
            val_loss = estimate_loss(model, validation_loader, device, validation_batches)
            print(f"step={step} train_loss={final_loss:.4f} val_loss={val_loss:.4f}")

    if not torch.isfinite(torch.tensor(final_loss)):
        raise RuntimeError("Training dataset produced no batches")

    if not args.smoke_test:
        torch.save(
            {
                "model": model.state_dict(),
                "config": vars(args),
                "tokenizer_name": getattr(tokenizer, "name_or_path", args.tokenizer_name),
            },
            args.checkpoint,
        )
    return model, final_loss


def main() -> None:
    """Parse arguments and run training or the offline end-to-end smoke test."""
    train(parse_args())


if __name__ == "__main__":
    main()
