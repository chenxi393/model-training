# -*- coding: utf-8 -*-
"""图片目录读取和 ResNet 输入预处理。"""

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
EXPECTED_CLASSES = {"normal": 0, "unsafe": 1}


def train_transform():
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def evaluation_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_datasets(root):
    root = Path(root)
    datasets_by_split = {
        "train": datasets.ImageFolder(root / "train", transform=train_transform()),
        "validation": datasets.ImageFolder(
            root / "validation", transform=evaluation_transform()
        ),
        "test": datasets.ImageFolder(root / "test", transform=evaluation_transform()),
    }
    for split, dataset in datasets_by_split.items():
        if dataset.class_to_idx != EXPECTED_CLASSES:
            raise ValueError(
                f"{split} 类别必须是 normal/unsafe，实际是 {dataset.class_to_idx}"
            )
    return datasets_by_split


def load_data(root, batch_size=16, num_workers=0):
    datasets_by_split = load_datasets(root)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
        )
        for split, dataset in datasets_by_split.items()
    }
    return datasets_by_split, loaders
