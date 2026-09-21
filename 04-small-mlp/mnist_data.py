# -*- coding: utf-8 -*-
"""下载并读取 MNIST 原始文件，不依赖 torchvision。"""

import gzip
import struct
import urllib.request
from pathlib import Path

import torch


BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def download_if_missing(root):
    root.mkdir(parents=True, exist_ok=True)
    for filename in FILES.values():
        path = root / filename
        if path.exists():
            continue
        url = BASE_URL + filename
        print(f"下载 {filename} ...")
        urllib.request.urlretrieve(url, path)


def read_images(path):
    with gzip.open(path, "rb") as f:
        magic, count, rows, columns = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"不是 MNIST 图片文件：{path}")
        data = f.read()
    images = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    return images.reshape(count, rows * columns).float() / 255.0


def read_labels(path):
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"不是 MNIST 标签文件：{path}")
        data = f.read()
    labels = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    return labels[:count].long()


def load_mnist(root="04-small-mlp/.data"):
    root = Path(root)
    download_if_missing(root)
    return (
        read_images(root / FILES["train_images"]),
        read_labels(root / FILES["train_labels"]),
        read_images(root / FILES["test_images"]),
        read_labels(root / FILES["test_labels"]),
    )
