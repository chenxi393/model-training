# -*- coding: utf-8 -*-
"""训练 normal / unsafe ResNet18，并保存 validation F1 最好的模型。"""

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn

from data import load_data
from metrics import binary_metrics, choose_threshold
from model import DEFAULT_CHECKPOINT, create_model, load_model, save_checkpoint


DEFAULT_DATA = Path(__file__).parent / ".data" / "demo"


def choose_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, loader, criterion, optimizer, device, frozen_backbone):
    if frozen_backbone:
        model.eval()
        model.fc.train()
    else:
        model.train()

    total_loss = 0.0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
    return total_loss / max(total, 1)


def evaluate(model, loader, criterion, device, unsafe_index):
    model.eval()
    total_loss = 0.0
    total = 0
    all_probabilities = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            probabilities = torch.softmax(logits, dim=1)[:, unsafe_index]
            all_probabilities.append(probabilities.cpu())
            all_labels.append(labels.cpu())
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)

    return (
        total_loss / max(total, 1),
        torch.cat(all_probabilities),
        torch.cat(all_labels),
    )


def metrics_line(metrics):
    return (
        f"threshold={metrics['threshold']:.2f} "
        f"precision={metrics['precision']:.3f} "
        f"recall={metrics['recall']:.3f} f1={metrics['f1']:.3f}"
    )


def main():
    parser = argparse.ArgumentParser(description="训练 normal / unsafe 图片分类模型")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--unfreeze", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(42)
    random.seed(42)
    device = choose_device(args.device)
    frozen_backbone = not args.unfreeze

    datasets_by_split, loaders = load_data(args.data, batch_size=args.batch_size)
    class_to_idx = datasets_by_split["train"].class_to_idx
    unsafe_index = class_to_idx["unsafe"]

    model = create_model(
        pretrained=not args.no_pretrained,
        freeze_backbone=frozen_backbone,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(trainable_parameters, lr=args.learning_rate)

    print(f"device={device} frozen_backbone={frozen_backbone}")
    print(
        " ".join(
            f"{split}={len(dataset)}" for split, dataset in datasets_by_split.items()
        )
    )
    print("epoch  train_loss  validation_loss  validation_metrics")

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            frozen_backbone,
        )
        validation_loss, probabilities, labels = evaluate(
            model,
            loaders["validation"],
            criterion,
            device,
            unsafe_index,
        )
        validation_metrics = choose_threshold(probabilities, labels)
        print(
            f"{epoch:>5}  {train_loss:>10.4f}  {validation_loss:>15.4f}  "
            f"{metrics_line(validation_metrics)}"
        )

        if validation_metrics["f1"] > best_f1:
            best_f1 = validation_metrics["f1"]
            save_checkpoint(
                model,
                args.checkpoint,
                class_to_idx=class_to_idx,
                threshold=validation_metrics["threshold"],
                validation_metrics=validation_metrics,
                best_epoch=epoch,
                backbone="resnet18",
                pretrained=not args.no_pretrained,
                frozen_backbone=frozen_backbone,
            )

    # 所有选择结束后，才加载最佳 checkpoint 并查看一次 test。
    best_model, metadata = load_model(args.checkpoint, device=device)
    test_loss, probabilities, labels = evaluate(
        best_model,
        loaders["test"],
        criterion,
        device,
        unsafe_index,
    )
    test_metrics = binary_metrics(probabilities, labels, metadata["threshold"])
    metadata["test_loss"] = test_loss
    metadata["test_metrics"] = test_metrics
    save_checkpoint(best_model, args.checkpoint, **metadata)

    print(f"\nbest_epoch={metadata['best_epoch']}")
    print(f"test_loss={test_loss:.4f} {metrics_line(test_metrics)}")
    print(
        f"TP={test_metrics['tp']} TN={test_metrics['tn']} "
        f"FP={test_metrics['fp']} FN={test_metrics['fn']}"
    )
    print(f"checkpoint={args.checkpoint}")


if __name__ == "__main__":
    main()
