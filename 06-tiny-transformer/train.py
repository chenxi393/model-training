# -*- coding: utf-8 -*-
"""训练一个字符级 tiny Transformer。"""

import argparse
import random
from pathlib import Path

import torch

from data import CharTokenizer, load_corpus, sample_batch, split_token_ids
from model import (
    DEFAULT_CHECKPOINT,
    TinyTransformer,
    TransformerConfig,
    save_checkpoint,
)


def choose_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def estimate_loss(model, datasets, batch_size, context_length, device, batches=20):
    model.eval()
    result = {}
    for split, data in datasets.items():
        losses = []
        for _ in range(batches):
            inputs, targets = sample_batch(
                data,
                batch_size,
                context_length,
                device,
            )
            _, loss, _ = model(inputs, targets)
            losses.append(loss.item())
        result[split] = sum(losses) / len(losses)
    model.train()
    return result


def main():
    parser = argparse.ArgumentParser(description="训练 tiny character Transformer")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path(__file__).parent / "tiny_corpus.txt",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=48)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = choose_device(args.device)

    text = load_corpus(args.corpus)
    tokenizer = CharTokenizer.from_text(text)
    token_ids = tokenizer.encode(text)
    train_ids, validation_ids = split_token_ids(token_ids)
    datasets = {"train": train_ids, "validation": validation_ids}

    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model = TinyTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"device={device} characters={len(text)} vocab={tokenizer.vocab_size} "
        f"parameters={parameter_count:,}"
    )
    print(
        f"train_tokens={len(train_ids)} validation_tokens={len(validation_ids)} "
        f"context={args.context_length}"
    )

    best_validation_loss = float("inf")
    best_step = 0
    model.train()
    for step in range(1, args.steps + 1):
        inputs, targets = sample_batch(
            train_ids,
            args.batch_size,
            args.context_length,
            device,
        )
        _, loss, _ = model(inputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        should_evaluate = step == 1 or step % args.eval_interval == 0
        if should_evaluate or step == args.steps:
            losses = estimate_loss(
                model,
                datasets,
                args.batch_size,
                args.context_length,
                device,
            )
            print(
                f"step={step:>4} train_loss={losses['train']:.4f} "
                f"validation_loss={losses['validation']:.4f}"
            )
            if losses["validation"] < best_validation_loss:
                best_validation_loss = losses["validation"]
                best_step = step
                save_checkpoint(
                    model,
                    tokenizer,
                    args.checkpoint,
                    best_step=best_step,
                    train_loss=losses["train"],
                    validation_loss=losses["validation"],
                    corpus=str(args.corpus),
                )

    print(f"best_step={best_step} validation_loss={best_validation_loss:.4f}")
    print(f"checkpoint={args.checkpoint}")


if __name__ == "__main__":
    main()
