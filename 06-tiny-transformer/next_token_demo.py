# -*- coding: utf-8 -*-
"""不用 PyTorch，先看懂语言模型到底在做哪一道题。"""


def main():
    text = "模型学习。"

    print("我们手里有一句训练文本：")
    print(f"  {text}")
    print()
    print("把每个字符的右边一个字符当作正确答案：")

    for position in range(1, len(text)):
        visible_text = text[:position]
        correct_next_character = text[position]
        print(
            f"  第 {position} 道题：看到 {visible_text!r}"
            f"，正确的下一个字符是 {correct_next_character!r}"
        )

    print()
    print("模型训练时，就是反复做这种“猜下一个字符”的题。")
    print("现在还没有 Tensor、Embedding 或 Attention，先确认这件事能看懂。")


if __name__ == "__main__":
    main()
