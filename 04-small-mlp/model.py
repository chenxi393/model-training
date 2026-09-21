# -*- coding: utf-8 -*-
"""MNIST MLP 的模型定义和 checkpoint 读写。"""

from pathlib import Path

import torch
import torch.nn as nn


CHECKPOINT_PATH = Path(__file__).with_name("prediction-demo") / "mnist_mlp.pt"


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.network(x)


def save_checkpoint(model, path=CHECKPOINT_PATH, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
        },
        path,
    )
    return path


def load_model(path=CHECKPOINT_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"找不到模型文件：{path}\n"
            "请先运行：uv run python 04-small-mlp/train_mlp.py"
        )

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = MLP()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint.get("metadata", {})
