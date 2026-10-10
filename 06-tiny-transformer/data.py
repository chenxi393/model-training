# -*- coding: utf-8 -*-
"""字符 tokenizer 和 next-token 训练数据。"""

from pathlib import Path

import torch


DEFAULT_CORPUS = Path(__file__).parent / "tiny_corpus.txt"


class CharTokenizer:
    def __init__(self, characters):
        self.characters = list(characters)
        self.char_to_id = {
            character: index for index, character in enumerate(self.characters)
        }

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.characters)

    def encode(self, text):
        unknown = sorted(set(text) - set(self.char_to_id))
        if unknown:
            raise ValueError(f"这些字符不在词表中：{unknown}")
        return [self.char_to_id[character] for character in text]

    def decode(self, token_ids):
        return "".join(self.characters[int(token_id)] for token_id in token_ids)


def load_corpus(path=DEFAULT_CORPUS):
    text = Path(path).read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"语料不能为空：{path}")
    return text


def split_token_ids(token_ids, train_fraction=0.9):
    split_index = int(len(token_ids) * train_fraction)
    train_ids = torch.tensor(token_ids[:split_index], dtype=torch.long)
    validation_ids = torch.tensor(token_ids[split_index:], dtype=torch.long)
    return train_ids, validation_ids


def sample_batch(data, batch_size, context_length, device, generator=None):
    max_start = len(data) - context_length - 1
    if max_start < 0:
        raise ValueError(
            f"数据只有 {len(data)} 个 token，无法生成长度为 {context_length} 的样本"
        )

    starts = torch.randint(
        0,
        max_start + 1,
        (batch_size,),
        generator=generator,
    )
    inputs = torch.stack([data[start : start + context_length] for start in starts])
    targets = torch.stack(
        [data[start + 1 : start + context_length + 1] for start in starts]
    )
    return inputs.to(device), targets.to(device)
