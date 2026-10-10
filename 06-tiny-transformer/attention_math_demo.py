# -*- coding: utf-8 -*-
"""用普通 Python 算一次 Attention 的三种基础运算。"""

import math


def dot(left, right):
    return sum(a * b for a, b in zip(left, right, strict=True))


def softmax(scores):
    maximum = max(scores)
    exponentials = [math.exp(score - maximum) for score in scores]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def main():
    query = [1, 2]
    keys = [[1, 0], [0, 1], [1, 1]]
    values = [[2, 0], [0, 2], [1, 1]]
    print("本例所有输入向量由我们手工设置，只用于观察计算。")
    print("1. 点积：相同位置的数相乘，再相加")
    scores = []
    for key in keys:
        score = dot(query, key)
        scores.append(score)
        print(f"  {query} · {key} = 1×{key[0]} + 2×{key[1]} = {score}")

    print("\n2. softmax：把分数转换成总和为 1 的权重")
    weights = softmax(scores)
    print(f"  原始分数：{scores}")
    print(f"  权重：{[round(weight, 3) for weight in weights]}")
    print(f"  权重之和：{sum(weights):.6f}")

    print("\n3. 加权求和：每个 Value 乘对应的权重，再相加")
    result = [0.0, 0.0]
    for weight, value in zip(weights, values, strict=True):
        contribution = [weight * number for number in value]
        result = [a + b for a, b in zip(result, contribution, strict=True)]
        print(f"  {weight:.3f} × {value} = {[round(v, 3) for v in contribution]}")
    print(f"  加起来：{[round(number, 3) for number in result]}")
    print("  这是读取到的信息向量，还不是下一个字符的概率。")
    print("\n完整 Attention 还会先将点积分数除以 √head_size，并遮住未来。")
    print("本练习先单独看清这三个运算，完整版本在 HTML 与 model.py 中。")


if __name__ == "__main__":
    main()
