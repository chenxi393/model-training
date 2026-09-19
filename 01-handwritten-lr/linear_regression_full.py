# -*- coding: utf-8 -*-
"""
线性回归 · 完整版

配套讲义：README.md 第 8 ~ 9 章

和 linear_regression.py 的区别只有两点：
    1. 多了一个参数 b  ->  y = w·x + b
    2. 点从一个变成 25 个  ->  斜率取【平均】

核心流程还是那 4 行。

跑：python3 linear_regression_full.py
"""
import numpy as np

from ascii_plot import scatter_with_line, plot_series

# ---- 数据：25 个带噪声的点。真相是 y = 2x + 1 ----------------------
# 看看模型能不能自己把 2 和 1 找回来。
rng = np.random.default_rng(42)
x = rng.uniform(0.0, 2.0, 25)
y = 2.0 * x + 1.0 + rng.normal(0.0, 0.15, 25)
TRUE_W, TRUE_B = 2.0, 1.0

# ---- 参数：现在是两个 --------------------------------------------
w, b = 0.0, 0.0

# ---- 你自己定的两个数 --------------------------------------------
LR = 0.1       # 学习率
STEPS = 150    # 挪多少次


# ---- 脚手架 ------------------------------------------------------

def predict(w, b):
    return w * x + b


def loss(w, b):
    """25 个点的误差平方，取平均"""
    return np.mean((predict(w, b) - y) ** 2)


def gradient(w, b):
    """
    每个参数各自的斜率。讲义第 8 章：

        w 的斜率 = 2 × 平均( 误差 × x )
        b 的斜率 = 2 × 平均( 误差     )

    为什么 b 那个不乘 x？因为 b 管整体上下平移，和 x 没关系。
    """
    error = predict(w, b) - y
    dw = np.mean(2 * error * x)
    db = np.mean(2 * error)
    return dw, db


# ==================================================================
#                        核心流程
# ==================================================================
loss_history = []

for step in range(STEPS):
    L = loss(w, b)                 # 1. 算 loss
    loss_history.append(L)
    dw, db = gradient(w, b)        # 2. 算梯度（两个斜率）
    w = w - LR * dw                # 3. 各自往反方向挪一小步
    b = b - LR * db                # 4.
# ==================================================================

print(f"模型学出   y = {w:.4f} x + {b:.4f}")
print(f"真实规律   y = {TRUE_W:.4f} x + {TRUE_B:.4f}")
print(f"最终 loss  {loss(w, b):.6f}\n")

# ---- 看一眼结果 --------------------------------------------------
scatter_with_line(
    x, y, TRUE_W, TRUE_B,
    title="  真实规律（* 是数据点，- 是真相）",
)
scatter_with_line(
    x, y, w, b,
    title="  模型自己学出来的（* 是数据点，- 是模型）",
)
plot_series(loss_history, title="  loss 一路怎么降下来的")
