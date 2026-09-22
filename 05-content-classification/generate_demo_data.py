# -*- coding: utf-8 -*-
"""生成安全的 normal / unsafe 流程演示图片。"""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


SPLIT_SIZES = {"train": 60, "validation": 20, "test": 20}
IMAGE_SIZE = 160


def random_color(rng, palette):
    base = rng.choice(palette)
    return tuple(max(0, min(255, value + rng.randint(-18, 18))) for value in base)


def draw_normal(rng, hard=False):
    cool = [(45, 130, 190), (50, 155, 110), (70, 120, 180)]
    warm = [(185, 95, 55), (175, 120, 45)]
    image = Image.new(
        "RGB",
        (IMAGE_SIZE, IMAGE_SIZE),
        random_color(rng, warm if hard else cool),
    )
    draw = ImageDraw.Draw(image)

    for _ in range(rng.randint(3, 6)):
        size = rng.randint(22, 55)
        x = rng.randint(5, IMAGE_SIZE - size - 5)
        y = rng.randint(5, IMAGE_SIZE - size - 5)
        color = random_color(rng, [(210, 235, 245), (180, 225, 195), (245, 240, 190)])
        draw.ellipse((x, y, x + size, y + size), fill=color)

    return image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 0.7)))


def draw_unsafe(rng, hard=False):
    warm = [(185, 55, 55), (210, 95, 35), (165, 65, 80)]
    cool = [(45, 125, 175), (50, 145, 105)]
    image = Image.new(
        "RGB",
        (IMAGE_SIZE, IMAGE_SIZE),
        random_color(rng, cool if hard else warm),
    )
    draw = ImageDraw.Draw(image)

    center_x = rng.randint(58, 102)
    center_y = rng.randint(58, 102)
    radius = rng.randint(38, 56)
    points = [
        (center_x, center_y - radius),
        (center_x - radius, center_y + radius),
        (center_x + radius, center_y + radius),
    ]
    draw.polygon(points, fill=random_color(rng, [(245, 210, 55), (245, 225, 95)]))
    draw.line((center_x, center_y - 18, center_x, center_y + 24), fill=(45, 35, 35), width=10)
    draw.ellipse((center_x - 5, center_y + 34, center_x + 5, center_y + 44), fill=(45, 35, 35))

    return image.rotate(rng.uniform(-8, 8), resample=Image.Resampling.BILINEAR)


def generate_dataset(root, seed=42):
    rng = random.Random(seed)
    for split, count_per_class in SPLIT_SIZES.items():
        for label, draw_image in (("normal", draw_normal), ("unsafe", draw_unsafe)):
            directory = root / split / label
            directory.mkdir(parents=True, exist_ok=True)
            for old_image in directory.glob("*.png"):
                old_image.unlink()

            for index in range(count_per_class):
                # validation/test 中放少量反常颜色，用来产生有意义的 Bad Case。
                hard = split != "train" and index < max(2, count_per_class // 5)
                image = draw_image(rng, hard=hard)
                image.save(directory / f"{label}_{index:03d}.png")


def main():
    parser = argparse.ArgumentParser(description="生成内容分类流程演示数据")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / ".data" / "demo",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_dataset(args.output, seed=args.seed)
    print(f"演示数据已生成：{args.output}")
    for split, count in SPLIT_SIZES.items():
        print(f"{split:>10}: normal={count} unsafe={count}")
    print("注意：这些是安全的合成图片，只用于跑通训练流程，不代表真实审核数据。")


if __name__ == "__main__":
    main()
