# -*- coding: utf-8 -*-
"""
终端里画图的小工具。

为什么不用 matplotlib？因为它要装。而这个文件里全是纯字符，
你把代码拷到任何一台机器上都能立刻跑起来看结果。
"""


def scatter_with_line(xs, ys, w, b, width=56, height=13, title=""):
    """把数据点(*)和模型直线(-)画在同一张 ASCII 图上。"""
    xs, ys = list(xs), list(ys)

    xmin, xmax = min(xs), max(xs)
    if xmax == xmin:
        xmax = xmin + 1.0

    # y 轴范围要把"直线"也算进去，否则线可能画到图外面
    ys_all = ys + [w * xmin + b, w * xmax + b]
    ymin, ymax = min(ys_all), max(ys_all)
    if ymax == ymin:
        ymax = ymin + 1.0
    pad = (ymax - ymin) * 0.08
    ymin, ymax = ymin - pad, ymax + pad

    grid = [[" "] * width for _ in range(height)]

    def cell(xv, yv):
        c = int(round((xv - xmin) / (xmax - xmin) * (width - 1)))
        r = int(round((ymax - yv) / (ymax - ymin) * (height - 1)))
        return max(0, min(height - 1, r)), max(0, min(width - 1, c))

    # 先画模型直线
    for i in range(width):
        xv = xmin + (xmax - xmin) * i / (width - 1)
        yv = w * xv + b
        if ymin <= yv <= ymax:
            r, c = cell(xv, yv)
            grid[r][c] = "-"

    # 再画真实数据点（后画的盖住先画的，所以点压在线上）
    for xv, yv in zip(xs, ys):
        r, c = cell(xv, yv)
        grid[r][c] = "*"

    if title:
        print(title)
    for r in range(height):
        yv = ymax - (ymax - ymin) * r / (height - 1)
        label = f"{yv:7.2f} |" if r in (0, height // 2, height - 1) else " " * 7 + " |"
        print(label + "".join(grid[r]))
    print(" " * 8 + "+" + "-" * width)
    left, right = f"{xmin:.2f}", f"{xmax:.2f}"
    print(" " * 8 + left + " " * max(1, width - len(left) - len(right)) + right)
    print()


def plot_series(series, width=56, height=8, title=""):
    """画一条随步数变化的曲线（比如 loss 曲线）。"""
    series = list(series)
    lo, hi = min(series), max(series)
    if hi == lo:
        hi = lo + 1.0

    grid = [[" "] * width for _ in range(height)]
    n = len(series)
    for i, v in enumerate(series):
        c = int(round(i / max(1, n - 1) * (width - 1)))
        r = int(round((hi - v) / (hi - lo) * (height - 1)))
        r = max(0, min(height - 1, r))
        grid[r][c] = "*"

    if title:
        print(title)
    for r in range(height):
        yv = hi - (hi - lo) * r / (height - 1)
        label = f"{yv:9.3f} |" if r in (0, height - 1) else " " * 9 + " |"
        print(label + "".join(grid[r]))
    print(" " * 10 + "+" + "-" * width)
    print(" " * 10 + f"step 0{' ' * max(1, width - 14)}step {n - 1}")
    print()
