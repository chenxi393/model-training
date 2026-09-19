# -*- coding: utf-8 -*-
"""
为什么每轮都要 zero_grad()

定位：课后调试实验，不属于 README.md 的 30 分钟主线。

backward() 往 .grad 里写的时候是【累加】，不是【覆盖】。
所以忘了清零，梯度就会把前几轮的旧账一起背在身上，
步子里混进了历史 —— 参数会冲过答案。

这个脚本跑两遍同一个循环，唯一的区别是有没有清零。

跑：python3 zero_grad_demo.py
"""
import torch

x = torch.tensor(2.0)
y = torch.tensor(6.0)      # 真相是 w = 3

LR = 0.05
STEPS = 5


def run(clear):
    """clear=True 就是正常写法；False 就是忘了写 zero_grad()"""
    w = torch.tensor(0.0, requires_grad=True)
    rows = []
    for step in range(STEPS):
        if clear and w.grad is not None:
            w.grad.zero_()                      # ← 差别只有这一行

        loss = (w * x - y) ** 2
        loss.backward()                         # 写进 w.grad 时是 +=

        w_now = w.item()
        g_seen = w.grad.item()                  # .grad 里现在攒了多少

        with torch.no_grad():
            w -= LR * g_seen

        # 这一步"真正"的梯度，用来对照 .grad 里是不是多了
        g_true = 2 * (w_now * x.item() - y.item()) * x.item()
        rows.append((step + 1, w_now, g_true, g_seen, w.item()))
    return rows


def show(rows, title):
    print(title)
    print(f"  {'step':>4}  {'更新前 w':>10}  {'本步真实梯度':>12}  {'.grad 里是':>11}  {'更新后 w':>10}")
    for step, w_now, g_true, g_seen, w_next in rows:
        flag = "   ← 多了旧账" if abs(g_seen - g_true) > 1e-6 else ""
        print(f"  {step:>4}  {w_now:>10.4f}  {g_true:>12.4f}  {g_seen:>11.4f}  {w_next:>10.4f}{flag}")
    print()


print("同一个循环，同一个起点，真相是 w = 3\n")
show(run(True), "【每轮清零】.grad 里只有本步的梯度：")
show(run(False), "【忘记清零】.grad 把前几轮的梯度一起背上了：")

print("""看出来了吗：

  清零那组，5 步就稳稳逼近 3。
  不清零那组，第 2 步的 .grad 就已经是 -38.4 而不是 -14.4 ——
  它把第 1 步的 -24 也加进来了。步子因此大得离谱，直接冲过答案。

  累加本身不是 bug，是设计。有些高级技巧（比如把好几批数据的
  梯度攒起来再更新一次）正要靠它。但对最普通的训练循环来说，
  上一轮的梯度是【过期的情报】—— 参数都挪过了，那笔账早就不算数了。

  所以 PyTorch 不替你清，你得自己写 optimizer.zero_grad()。
""")
