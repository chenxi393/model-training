# -*- coding: utf-8 -*-
"""
PyTorch 线性回归 · 完整版

配套讲义：README.md 第 5 ~ 8 章

和第 1 单元的 linear_regression_full.py 干的是同一件事：25 个点，两个参数。
区别是这次不用自己写 predict / loss / gradient 了 ——
nn.Module 管参数，nn.MSELoss 管 loss，backward 管梯度，optimizer 管更新。

核心还是那 5 行，一行不多。

跑：python3 pytorch_regression_full.py
"""
import torch
import torch.nn as nn

from data import X, Y, TRUE_W, TRUE_B
from ascii_plot import scatter_with_line, plot_series

# ---- 数据：25 个带噪声的点。真相是 y = 2x + 1 ----------------------
# nn.Linear 要的是 (样本数, 特征数)，所以 x、y 都竖过来，变成 25 行 1 列。
x = torch.tensor(X, dtype=torch.float32).reshape(-1, 1)
y = torch.tensor(Y, dtype=torch.float32).reshape(-1, 1)

# ---- 你自己定的两个数 --------------------------------------------
LR = 0.1       # 学习率
STEPS = 150    # 挪多少次


# ---- 模型：一个盒子，里面装着一个 w 和一个 b ----------------------
torch.manual_seed(42)          # nn.Linear 的 w、b 是随机的，先固定住


class Line(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)     # 1 个输入，1 个输出

    def forward(self, x):
        return self.linear(x)             # 盒子里做的就是 w·x + b


model = Line()
criterion = nn.MSELoss()                                  # 就是第 1 单元第 2 章那个 loss
optimizer = torch.optim.SGD(model.parameters(), lr=LR)    # 就是第 1 单元第 7 章那个 w -= lr·g


# ==================================================================
#                        核心流程
# ==================================================================
loss_history = []

for step in range(STEPS):
    optimizer.zero_grad()              # 1. 清掉上一轮留下的梯度（第 4 章）
    y_pred = model(x)                  # 2. 前向：算出预测
    loss = criterion(y_pred, y)        # 3. 算 loss：这次错了多重
    loss.backward()                    # 4. 算梯度：每个参数各自的斜率
    optimizer.step()                   # 5. 往反方向挪一小步

    loss_history.append(loss.item())
# ==================================================================


w = model.linear.weight.item()
b = model.linear.bias.item()

print(f"模型学出   y = {w:.4f} x + {b:.4f}")
print(f"真实规律   y = {TRUE_W:.4f} x + {TRUE_B:.4f}")
print(f"最终 loss  {loss_history[-1]:.6f}\n")

# ---- 看一眼结果 --------------------------------------------------
scatter_with_line(
    X, Y, TRUE_W, TRUE_B,
    title="  真实规律（* 是数据点，- 是真相）",
)
scatter_with_line(
    X, Y, w, b,
    title="  模型自己学出来的（* 是数据点，- 是模型）",
)
plot_series(loss_history, title="  loss 一路怎么降下来的")
