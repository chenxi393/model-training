# -*- coding: utf-8 -*-
"""使用训练好的 tiny Transformer 逐字生成文本。"""

import argparse
from pathlib import Path

import torch

from model import DEFAULT_CHECKPOINT, load_checkpoint
from train import choose_device


def main():
    parser = argparse.ArgumentParser(description="使用 tiny Transformer 生成文字")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--prompt", default="模型")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    model, tokenizer, metadata = load_checkpoint(args.checkpoint, device=device)
    prompt_ids = torch.tensor(
        [tokenizer.encode(args.prompt)],
        dtype=torch.long,
        device=device,
    )
    generated_ids = model.generate(
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    print(f"device={device} best_step={metadata.get('best_step')}")
    print(tokenizer.decode(generated_ids[0].tolist()))


if __name__ == "__main__":
    main()
