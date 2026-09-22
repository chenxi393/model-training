# -*- coding: utf-8 -*-
"""使用训练好的内容分类模型预测一张图片。"""

import argparse
from pathlib import Path

from inference import open_image, predict_pil
from model import DEFAULT_CHECKPOINT, load_model
from train import choose_device


def main():
    parser = argparse.ArgumentParser(description="预测一张 normal / unsafe 图片")
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    model, metadata = load_model(args.checkpoint, device=device)
    result = predict_pil(model, open_image(args.image), metadata, device=device)

    print(f"图片：{args.image}")
    print(f"normal：{result['probabilities']['normal']:.2%}")
    print(f"unsafe：{result['probabilities']['unsafe']:.2%}")
    print(f"threshold：{result['threshold']:.2f}")
    print(f"最终判断：{result['decision']}")


if __name__ == "__main__":
    main()
