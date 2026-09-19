# -*- coding: utf-8 -*-
"""
用 PyTorch 重写最小线性回归。

学习主线：Tensor -> nn.Module -> loss.backward() -> optimizer.step()

运行：
    uv run python 02-pytorch-lr/pytorch_regression.py
"""

import torch
import torch.nn as nn


# 一个训练样本，输入和目标都写成“1 行 1 列”。
x = torch.tensor([[2.0]])
y = torch.tensor([[6.0]])


class Line(nn.Module):
    """只有一个参数 w 的模型：y_pred = w * x。"""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        return self.linear(x)


model = Line()
with torch.no_grad():
    model.linear.weight.zero_()

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

steps = 12
first_step = None

for step in range(steps):
    optimizer.zero_grad()          # 清掉上一轮梯度
    y_pred = model(x)              # nn.Module 算预测
    loss = criterion(y_pred, y)    # 算预测与真实值的差距
    loss.backward()                # 把梯度写入 model 的参数 .grad

    if step == 0:
        first_step = {
            "y_pred": y_pred.item(),
            "loss": loss.item(),
            "grad": model.linear.weight.grad.item(),
        }

    optimizer.step()               # 对所有参数执行 parameter -= lr * grad

    if step == 0:
        first_step["weight_after"] = model.linear.weight.item()


with torch.no_grad():
    final_loss = criterion(model(x), y).item()

print("第一轮发生了什么：")
print(f"  y_pred       = {first_step['y_pred']:.1f}")
print(f"  loss         = {first_step['loss']:.1f}")
print(f"  weight.grad  = {first_step['grad']:.1f}")
print(f"  更新后 weight = {first_step['weight_after']:.1f}\n")
print(f"学到的 weight = {model.linear.weight.item():.4f}")
print("真实是 weight = 3")
print(f"最终 loss = {final_loss:.6f}")
