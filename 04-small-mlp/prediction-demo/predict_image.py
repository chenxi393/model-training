# -*- coding: utf-8 -*-
"""加载训练好的 MLP，预测一张手写数字图片。"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image, ImageOps, ImageStat


CHAPTER_DIR = Path(__file__).resolve().parent.parent
if str(CHAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(CHAPTER_DIR))

from model import CHECKPOINT_PATH, load_model


def normalize_image(image):
    """把 Pillow 图片转换成接近 MNIST 的黑底白字 28x28 图片。"""
    image = image.convert("L")
    # MNIST 是黑底白字。常见手写图片是白底黑字，因此按平均亮度自动反色。
    if ImageStat.Stat(image).mean[0] > 127:
        image = ImageOps.invert(image)

    # 导出的测试图片已经和训练数据同为 28x28，不再重复裁剪和缩放。
    if image.size == (28, 28):
        return image

    image = ImageOps.autocontrast(image)
    bounding_box = image.getbbox()
    if bounding_box is None:
        raise ValueError("图片是空白的，没有找到手写数字")

    digit = image.crop(bounding_box)
    digit.thumbnail((20, 20), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), color=0)
    left = (28 - digit.width) // 2
    top = (28 - digit.height) // 2
    canvas.paste(digit, (left, top))
    return canvas


def image_to_tensor(image):
    pixels = torch.tensor(
        list(image.get_flattened_data()), dtype=torch.float32
    ) / 255.0
    return pixels.reshape(1, 28 * 28)


def prepare_image(path):
    """读取图片并返回可直接交给 MLP 的 Tensor。"""
    with Image.open(path) as image:
        normalized = normalize_image(image)
    return image_to_tensor(normalized)


def predict(model, image_tensor):
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
    predicted_digit = int(probabilities.argmax().item())
    return predicted_digit, probabilities


def main():
    parser = argparse.ArgumentParser(description="预测一张手写数字图片")
    parser.add_argument("image", type=Path, help="PNG/JPG 等图片路径")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_PATH,
        help=f"模型文件路径，默认：{CHECKPOINT_PATH}",
    )
    args = parser.parse_args()

    model, metadata = load_model(args.checkpoint)
    image_tensor = prepare_image(args.image)
    predicted_digit, probabilities = predict(model, image_tensor)

    print(f"图片：{args.image}")
    if "test_accuracy" in metadata:
        print(f"checkpoint test_acc：{metadata['test_accuracy']:.3f}")
    print("\n数字  概率")
    for digit, probability in enumerate(probabilities.tolist()):
        marker = "  <-- 预测" if digit == predicted_digit else ""
        print(f"  {digit}   {probability:>7.2%}{marker}")


if __name__ == "__main__":
    main()
