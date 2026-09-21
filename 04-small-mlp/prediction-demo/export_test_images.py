# -*- coding: utf-8 -*-
"""从 MNIST test 中为数字 0-9 各导出一张 PNG。"""

import sys
from pathlib import Path

from PIL import Image


CHAPTER_DIR = Path(__file__).resolve().parent.parent
if str(CHAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(CHAPTER_DIR))

from mnist_data import load_mnist


OUTPUT_DIR = Path(__file__).with_name("test_images")


def main():
    _, _, test_x, test_y = load_mnist()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    exported = set()
    for image_tensor, label_tensor in zip(test_x, test_y):
        label = int(label_tensor.item())
        if label in exported:
            continue

        pixels = (image_tensor.reshape(28, 28) * 255).byte().numpy()
        path = OUTPUT_DIR / f"digit_{label}.png"
        Image.fromarray(pixels).save(path)
        print(f"导出 {path}")
        exported.add(label)

        if len(exported) == 10:
            break


if __name__ == "__main__":
    main()
