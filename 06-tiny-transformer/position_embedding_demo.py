# -*- coding: utf-8 -*-
"""观察同一个字符怎样因为位置不同而得到不同的最终表示。"""

import torch

from data import CharTokenizer


def rounded(vector):
    return [round(value, 3) for value in vector.detach().tolist()]


def main():
    torch.manual_seed(42)

    text = "模型模"
    tokenizer = CharTokenizer.from_text(text)
    input_ids = torch.tensor([tokenizer.encode(text)])
    positions = torch.arange(len(text))

    token_embedding = torch.nn.Embedding(tokenizer.vocab_size, 3)
    position_embedding = torch.nn.Embedding(len(text), 3)

    token_vectors = token_embedding(input_ids)[0]
    position_vectors = position_embedding(positions)
    final_vectors = token_vectors + position_vectors

    print(f"输入文字：{text!r}")
    print(f"字符 ID：{input_ids.tolist()}")
    print(f"位置 ID：{positions.tolist()}")
    print()
    print("第 1 个和第 3 个字符都是“模”，所以 Token Embedding 相同：")
    print(f"  第 1 个“模”：{rounded(token_vectors[0])}")
    print(f"  第 3 个“模”：{rounded(token_vectors[2])}")
    print()
    print("但它们的位置不同，所以 Position Embedding 不同：")
    print(f"  位置 0：{rounded(position_vectors[0])}")
    print(f"  位置 2：{rounded(position_vectors[2])}")
    print()
    print("两种向量相加后，同一个“模”在两个位置有了不同表示：")
    print(f"  第 1 个“模”：{rounded(final_vectors[0])}")
    print(f"  第 3 个“模”：{rounded(final_vectors[2])}")
    print()
    print("当前位置表示 = Token Embedding + Position Embedding")
    print("两张表开始都是随机值，之后都会通过训练更新。")


if __name__ == "__main__":
    main()
