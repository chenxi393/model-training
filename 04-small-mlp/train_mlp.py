# -*- coding: utf-8 -*-
"""在 MNIST 上训练一个小型 MLP。"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mnist_data import load_mnist
from model import MLP, save_checkpoint


torch.manual_seed(42)

TRAIN_SIZE = 10_000
VALIDATION_SIZE = 2_000
TEST_SIZE = 2_000
BATCH_SIZE = 64
EPOCHS = 8
LEARNING_RATE = 0.1


def accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            predictions = logits.argmax(dim=1)
            correct += (predictions == y).sum().item()
            total += y.numel()
    return correct / total


def evaluate_loss(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            loss = criterion(model(x), y)
            total_loss += loss.item() * y.size(0)
            total += y.size(0)
    return total_loss / total


def main():
    train_x, train_y, test_x, test_y = load_mnist()

    # 从训练集随机切出 validation。test 全程留到最后才使用。
    split_indices = torch.randperm(len(train_x))[: TRAIN_SIZE + VALIDATION_SIZE]
    train_x = train_x[split_indices]
    train_y = train_y[split_indices]
    validation_x = train_x[TRAIN_SIZE:]
    validation_y = train_y[TRAIN_SIZE:]
    train_x = train_x[:TRAIN_SIZE]
    train_y = train_y[:TRAIN_SIZE]

    test_x = test_x[:TEST_SIZE]
    test_y = test_y[:TEST_SIZE]

    train_loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=True
    )
    validation_loader = DataLoader(
        TensorDataset(validation_x, validation_y), batch_size=BATCH_SIZE
    )
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=BATCH_SIZE)

    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    print(
        f"train={len(train_x)}  validation={len(validation_x)}  test={len(test_x)}  "
        f"batch_size={BATCH_SIZE}  epochs={EPOCHS}\n"
    )
    print("epoch  train_loss  train_acc  validation_loss  validation_acc")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x_batch, y_batch in train_loader:
            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = evaluate_loss(model, train_loader, criterion)
        validation_loss = evaluate_loss(model, validation_loader, criterion)
        train_acc = accuracy(model, train_loader)
        validation_acc = accuracy(model, validation_loader)
        print(
            f"{epoch:>5}  {train_loss:>10.4f}  {train_acc:>9.3f}  "
            f"{validation_loss:>15.4f}  {validation_acc:>14.3f}"
        )

    test_accuracy = accuracy(model, test_loader)
    checkpoint_path = save_checkpoint(
        model,
        test_accuracy=test_accuracy,
        epochs=EPOCHS,
        train_size=TRAIN_SIZE,
        validation_size=VALIDATION_SIZE,
        test_size=TEST_SIZE,
    )
    print(f"\n最终 test_acc = {test_accuracy:.3f}")
    print(f"模型已保存：{checkpoint_path}")


if __name__ == "__main__":
    main()
