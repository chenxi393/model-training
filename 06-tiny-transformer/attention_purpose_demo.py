# -*- coding: utf-8 -*-
"""先理解 Attention 解决什么问题，不计算 Q、K、V。"""


def describe(text):
    last_character = text[-1]
    last_position = len(text) - 1
    return (
        f"字符 {last_character!r} 的 Token Embedding"
        f" + 位置 {last_position} 的 Position Embedding"
    )


def main():
    first_text = "我吃了"
    second_text = "天黑了"

    print("我们比较两段输入：")
    print(f"  A：{first_text}")
    print(f"  B：{second_text}")
    print()
    print("在 Attention 之前，只看最后一个位置：")
    print(f"  A 的最后位置 = {describe(first_text)}")
    print(f"  B 的最后位置 = {describe(second_text)}")
    print("  两者完全相同，因为字符都是“了”，位置也都是 2。")
    print()
    print("训练题 A 的答案是“饭”，训练题 B 的答案是“。”。")
    print("只看最后一个“了”无法区分这两道题，需要读取前文。")
    print()
    print("Attention 让最后一个位置读取当前可见的全部位置：")
    print(f"  A 的“了”可以读取：{list(first_text)}")
    print(f"  B 的“了”可以读取：{list(second_text)}")
    print()
    print("Attention 可以让 A 和 B 最后位置得到不同的上下文表示。")
    print("这就是 Attention 先要解决的问题。")
    print()
    print("下一步才研究：模型怎样决定每个位置应该读取多少。")


if __name__ == "__main__":
    main()
