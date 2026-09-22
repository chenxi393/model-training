# -*- coding: utf-8 -*-
"""ResNet18 迁移学习模型和 checkpoint 读写。"""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


DEFAULT_CHECKPOINT = Path(__file__).parent / "artifacts" / "content_resnet18.pt"


def create_model(pretrained=True, freeze_backbone=True, num_classes=2):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, num_classes)
    return model


def save_checkpoint(model, path=DEFAULT_CHECKPOINT, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "metadata": metadata},
        path,
    )
    return path


def load_model(path=DEFAULT_CHECKPOINT, device="cpu"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 checkpoint：{path}\n"
            "请先运行：uv run python 05-content-classification/train.py"
        )
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = create_model(pretrained=False, freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint.get("metadata", {})
