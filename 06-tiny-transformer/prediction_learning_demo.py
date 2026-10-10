# -*- coding: utf-8 -*-
"""让真实 Transformer 记住两道题，观察概率、loss 和参数变化。"""

import torch
import torch.nn.functional as F

from data import CharTokenizer
from model import TinyTransformer, TransformerConfig


@torch.no_grad()
def show_predictions(model, tokenizer, inputs, targets, prompts, label):
    model.eval()
    logits, _, _ = model(inputs)
    last_logits = logits[:, -1, :]
    probabilities = F.softmax(last_logits, dim=-1)
    losses = F.cross_entropy(last_logits, targets, reduction="none")
    print(f"\n{label}")
    for row, prompt in enumerate(prompts):
        target_id = targets[row].item()
        print(
            f"  输入={prompt!r} 正确答案={tokenizer.decode([target_id])!r} "
            f"正确答案logit={last_logits[row, target_id].item():.4f} "
            f"正确答案概率={probabilities[row, target_id].item():.4f} "
            f"loss={losses[row].item():.4f}"
        )
        top_probabilities, top_ids = probabilities[row].topk(3)
        candidates = [
            f"{tokenizer.decode([token_id])!r}:{probability:.3f}"
            for probability, token_id in zip(
                top_probabilities.tolist(), top_ids.tolist(), strict=True
            )
        ]
        print(f"    概率最高的 3 个候选：{', '.join(candidates)}")
    return losses.mean().item()


def main():
    torch.manual_seed(42)
    prompts = ["我吃了", "天黑了"]
    answers = ["饭", "。"]
    tokenizer = CharTokenizer.from_text("".join(prompts + answers))
    inputs = torch.tensor([tokenizer.encode(prompt) for prompt in prompts])
    targets = torch.tensor([tokenizer.encode(answer)[0] for answer in answers])
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=3,
        d_model=16,
        num_heads=1,
        num_layers=1,
        dropout=0.0,
    )
    model = TinyTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    token_id = tokenizer.encode("了")[0]
    embedding_before = model.token_embedding.weight[token_id].detach().clone()
    qkv_before = model.blocks[0].attention.qkv.weight.detach().clone()

    print("CPU 上训练真实 TinyTransformer，所有可训练参数都参与更新。")
    print("只有两道固定练习题；本例展示记住题目的过程，不衡量泛化能力。")
    before_loss = show_predictions(
        model, tokenizer, inputs, targets, prompts, "训练前"
    )

    model.train()
    for _ in range(150):
        logits, _, _ = model(inputs)
        # 每道题只监督最后位置，便于观察“同一个了，不同的下一个字”。
        loss = F.cross_entropy(logits[:, -1, :], targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    after_loss = show_predictions(
        model, tokenizer, inputs, targets, prompts, "训练 150 步后"
    )
    embedding_after = model.token_embedding.weight[token_id].detach()
    qkv_after = model.blocks[0].attention.qkv.weight.detach()
    print("\n训练实际修改了什么？")
    print(f"  '了' 的 ID：{token_id}（没变）")
    print(f"  Embedding 前 3 个数，训练前：{embedding_before[:3].tolist()}")
    print(f"  Embedding 前 3 个数，训练后：{embedding_after[:3].tolist()}")
    print(f"  Q/K/V 权重最大变化量：{(qkv_after - qkv_before).abs().max().item():.4f}")
    print(f"  两道题的平均 loss：{before_loss:.4f} → {after_loss:.4f}")
    print("  Embedding 是输入特征；输出 logits 经 softmax 才得到候选字符概率。")
    print("\n完整 train.py 会监督序列中的每个位置，并另用 validation 选模型。")


if __name__ == "__main__":
    main()
