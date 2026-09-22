# -*- coding: utf-8 -*-
"""把 Hugging Face NSFW Parquet 分片抽取成 05 可读取的图片目录。"""

import argparse
import hashlib
import io
import json
import shutil
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image, ImageOps, UnidentifiedImageError


LABEL_NAMES = {0: "unsafe", 1: "normal"}
SPLITS = (
    ("train", 80),
    ("validation", 90),
    ("test", 100),
)


def choose_split(digest):
    bucket = int(digest[:8], 16) % 100
    for name, upper_bound in SPLITS:
        if bucket < upper_bound:
            return name
    raise AssertionError("split bucket 必须在 0～99")


def image_bytes(value):
    if not isinstance(value, dict):
        raise ValueError("image 列不是预期的 {bytes, path} 结构")
    if value.get("bytes"):
        return value["bytes"]
    if value.get("path"):
        return Path(value["path"]).read_bytes()
    raise ValueError("image 记录里没有 bytes 或 path")


def export_parquet_files(
    parquet_files,
    output,
    extra_hf_files=(),
    per_class=None,
):
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    counts = Counter()
    label_counts = Counter()
    seen_digests = set()
    skipped = Counter()

    def targets_reached():
        return per_class is not None and all(
            label_counts[label] >= per_class for label in LABEL_NAMES.values()
        )

    def extract_parquet(parquet):
        for batch in parquet.iter_batches(batch_size=64, columns=["image", "label"]):
            images = batch.column("image").to_pylist()
            labels = batch.column("label").to_pylist()
            for image_value, label_value in zip(images, labels, strict=True):
                try:
                    label = LABEL_NAMES[int(label_value)]
                    if per_class is not None and label_counts[label] >= per_class:
                        continue

                    source_bytes = image_bytes(image_value)
                    digest = hashlib.sha256(source_bytes).hexdigest()
                    if digest in seen_digests:
                        skipped["duplicate"] += 1
                        continue

                    split = choose_split(digest)
                    destination = output / split / label / f"{digest}.jpg"
                    destination.parent.mkdir(parents=True, exist_ok=True)

                    with Image.open(io.BytesIO(source_bytes)) as image:
                        image = ImageOps.exif_transpose(image).convert("RGB")
                        image.save(destination, format="JPEG", quality=90, optimize=True)

                    seen_digests.add(digest)
                    label_counts[label] += 1
                    counts[f"{split}/{label}"] += 1
                    counts["total"] += 1
                except (KeyError, ValueError, OSError, UnidentifiedImageError):
                    skipped["invalid"] += 1

            if targets_reached():
                return

    source_names = []
    for parquet_path in parquet_files:
        source_names.append(str(parquet_path))
        extract_parquet(pq.ParquetFile(parquet_path))
        if targets_reached():
            break

    if extra_hf_files and not targets_reached():
        from huggingface_hub import HfFileSystem

        filesystem = HfFileSystem()
        for hf_path in extra_hf_files:
            source_names.append(f"hf://{hf_path}")
            with filesystem.open(hf_path, "rb") as remote_file:
                extract_parquet(pq.ParquetFile(remote_file))
            if targets_reached():
                break

    summary = {
        "source_files": source_names,
        "label_mapping": {"0": "unsafe", "1": "normal"},
        "split_rule": "图片 SHA-256 哈希后按 80%/10%/10% 划分",
        "per_class_target": per_class,
        "counts": dict(sorted(counts.items())),
        "skipped": dict(sorted(skipped.items())),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="抽取 NSFW Parquet 图片样本")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).parent / ".data" / "nsfw-sample" / "source" / "data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / ".data" / "nsfw-sample" / "images",
    )
    parser.add_argument(
        "--extra-hf-file",
        action="append",
        default=[],
        help="额外按需读取的 hf:// 文件路径，可重复传入",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        help="每个类别最多抽取多少张；不传则抽取全部本地记录",
    )
    args = parser.parse_args()

    parquet_files = sorted(args.source.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"没有找到 Parquet 分片：{args.source}")

    summary = export_parquet_files(
        parquet_files,
        args.output,
        extra_hf_files=args.extra_hf_file,
        per_class=args.per_class,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"图片目录：{args.output}")


if __name__ == "__main__":
    main()
