# -*- coding: utf-8 -*-
"""理解 Next Token Prediction 后，再检查 Transformer 内部的 Tensor 形状。"""

import torch

from data import CharTokenizer
from model import TinyTransformer, TransformerConfig


def main():
    text = "模型根据数据学习规律。"
    tokenizer = CharTokenizer.from_text(text)
    input_text = text[:-1]
    target_text = text[1:]
    input_ids = torch.tensor([tokenizer.encode(input_text)])
    targets = torch.tensor([tokenizer.encode(target_text)])

    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=len(input_text),
        d_model=16,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
    )
    model = TinyTransformer(config).eval()
    logits, loss, attentions = model(
        input_ids,
        targets,
        return_attention=True,
    )

    print("原始文本：", text)
    print("模型输入：", input_text)
    print("正确答案：", target_text)
    print("\n每个位置都在根据当前可见的前缀预测下一个字符：")
    for position, target in enumerate(target_text, start=1):
        visible_text = input_text[:position]
        print(f"  看到 {visible_text!r} → 应该预测 {target!r}")

    embedding_shape = tuple(model.token_embedding(input_ids).shape)
    head_size = config.d_model // config.num_heads
    print("\nTensor 形状：")
    print(f"  input_ids       {tuple(input_ids.shape)}  [batch, position]")
    print(f"  embedding       {embedding_shape}  [batch, position, d_model]")
    print(
        f"  query/key/value {(1, config.num_heads, len(input_text), head_size)}  "
        "[batch, head, position, head_size]"
    )
    print(
        f"  attention       {tuple(attentions[0].shape)}  "
        "[batch, head, query_position, key_position]"
    )
    print(
        f"  logits          {tuple(logits.shape)}  "
        "[batch, position, vocab_size]"
    )
    print(f"  loss            {loss.item():.4f}")

    print("\n第一个 attention head 的权重（未训练，所以数值没有语义）：")
    print(attentions[0][0, 0].detach().round(decimals=2))
    print("右上角全部为 0，表示当前位置看不到未来字符。")


if __name__ == "__main__":
    main()
