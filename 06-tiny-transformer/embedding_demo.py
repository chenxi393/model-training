# -*- coding: utf-8 -*-
"""只观察字符怎样经过 ID 查到 Embedding 向量。"""

import torch

from data import CharTokenizer


def main():
    torch.manual_seed(42)

    training_text = "模型学习。"
    input_text = "模型学"
    tokenizer = CharTokenizer.from_text(training_text)

    print("第一步：给每种字符分配一个 ID")
    for character in tokenizer.characters:
        print(f"  {character!r} → {tokenizer.char_to_id[character]}")

    input_ids = torch.tensor([tokenizer.encode(input_text)])
    print()
    print("第二步：把输入字符替换成 ID")
    print(f"  输入文字：{input_text!r}")
    print(f"  input_ids：{input_ids.tolist()}")
    print("  ID 只是 Embedding 表的行号，不是字符的特征。")

    embedding = torch.nn.Embedding(
        num_embeddings=tokenizer.vocab_size,
        embedding_dim=3,
    )
    vectors = embedding(input_ids)

    print()
    print("第三步：用每个 ID 查 Embedding 表的一行")
    for character, token_id, vector in zip(
        input_text,
        input_ids[0],
        vectors[0],
        strict=True,
    ):
        values = [round(value, 3) for value in vector.detach().tolist()]
        print(f"  {character!r} → 第 {token_id.item()} 行 → {values}")

    print()
    print(f"input_ids 形状：{tuple(input_ids.shape)}  [1 段文字, 3 个位置]")
    print(f"Embedding 形状：{tuple(vectors.shape)}  [1 段文字, 3 个位置, 3 个数]")
    print()
    print("现在这些小数是随机初始值，所以还没有语义。")
    print("训练时，loss.backward() 和 optimizer.step() 会逐渐修改这张表。")


if __name__ == "__main__":
    main()
