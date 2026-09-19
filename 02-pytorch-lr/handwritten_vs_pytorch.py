# -*- coding: utf-8 -*-
"""
手写 vs PyTorch：同一份数据，同一个起点，同一个学习率

定位：README.md 学完以后的结果验证。

左边是第 1 单元我们一行行推出来的东西，右边是 PyTorch。
如果 backward() 真的只是"自动帮你算斜率"，两边应该走出完全一样的路。

跑：python3 handwritten_vs_pytorch.py
"""
import torch
import torch.nn as nn

from data import X, Y, TRUE_W, TRUE_B

N = len(X)
LR = 0.1
STEPS = 150


# ==================================================================
# 左边：第 1 单元的手写版。梯度是我们自己推的公式。
# ==================================================================
def handwritten():
    w, b = 0.0, 0.0
    for _ in range(STEPS):
        e = [w * X[i] + b - Y[i] for i in range(N)]           # 误差
        dw = sum(2 * e[i] * X[i] for i in range(N)) / N        # w 的斜率
        db = sum(2 * e[i] for i in range(N)) / N               # b 的斜率
        w -= LR * dw
        b -= LR * db
    return w, b


# ==================================================================
# 右边：PyTorch 版。梯度交给 backward()。
# ==================================================================
def pytorch_version():
    x = torch.tensor(X, dtype=torch.float64).reshape(-1, 1)
    y = torch.tensor(Y, dtype=torch.float64).reshape(-1, 1)

    model = nn.Linear(1, 1, dtype=torch.float64)
    with torch.no_grad():            # 手写版从 0 出发，这边也从 0 出发，才好比
        model.weight.zero_()
        model.bias.zero_()

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    for _ in range(STEPS):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    return model.weight.item(), model.bias.item()


w_hand, b_hand = handwritten()
w_torch, b_torch = pytorch_version()


def mse(w, b):
    return sum((w * X[i] + b - Y[i]) ** 2 for i in range(N)) / N


print(f"数据 {N} 个点，学习率 {LR}，跑 {STEPS} 步，两个都从 w=0, b=0 出发\n")
print(f"  {'':<12}{'w':>18}{'b':>18}{'loss':>18}")
print(f"  {'手写公式':<10}{w_hand:>18.12f}{b_hand:>18.12f}{mse(w_hand, b_hand):>18.12f}")
print(f"  {'PyTorch':<11}{w_torch:>18.12f}{b_torch:>18.12f}{mse(w_torch, b_torch):>18.12f}")
print(f"  {'相差':<12}{abs(w_hand - w_torch):>18.2e}{abs(b_hand - b_torch):>18.2e}")
print()

gap = max(abs(w_hand - w_torch), abs(b_hand - b_torch))
if gap < 1e-9:
    print("""小数点后十位都一样。

  这不是"差不多"，是同一件事的两种写法。
  backward() 没有做任何我们没做过的事 —— 它只是把我们手推的那个公式
  （w 的斜率 = 2×平均(误差×x)，b 的斜率 = 2×平均(误差)）
  沿着计算图自动走了一遍。

  唯一的区别是：我们的公式只能用在"一条直线"上。
  换个模型，我们得重新推一遍；它不用。
""")
else:
    print(f"两边差在 {gap:.2e} 这个量级 —— 是浮点数加法顺序不同造成的舍入误差，不是算法不同。")
