# 用 PyTorch 重写线性回归

预计学习时间：30 分钟。

这次只理解一条主线：

```text
Tensor
→ nn.Module
→ loss.backward()
→ optimizer.step()
```

学完后，你应该能说清楚 PyTorch 自动帮我们做了什么，以及哪些事情仍然由我们决定。

配套代码：[pytorch_regression.py](./pytorch_regression.py)

---

## 0～5 分钟：先运行

在仓库根目录执行：

```bash
uv run python 02-pytorch-lr/pytorch_regression.py
```

你会看到：

```text
第一轮发生了什么：
  y_pred       = 0.0
  loss         = 36.0
  weight.grad  = -24.0
  更新后 weight = 1.2

学到的 weight = 2.9935
真实是 weight = 3
```

模型从 `weight = 0` 开始。训练 12 次以后，weight 接近正确答案 `3`。

接下来只看这 5 行训练代码：

```python
optimizer.zero_grad()
y_pred = model(x)
loss = criterion(y_pred, y)
loss.backward()
optimizer.step()
```

---

## 5～10 分钟：Tensor 装数据

代码中的训练数据是：

```python
x = torch.tensor([[2.0]])
y = torch.tensor([[6.0]])
```

它表示一个训练样本：

```text
输入 x = 2
真实答案 y = 6
```

Tensor 可以理解为 PyTorch 使用的数组。这里写成 `[[2.0]]`，表示“1 个样本，每个样本有 1 个特征”。

| Tensor | 形状 | 含义 |
|---|---|---|
| `x` | `(1, 1)` | 1 个样本，1 个输入特征 |
| `y` | `(1, 1)` | 这个样本对应 1 个真实答案 |

Tensor 在这里做两件事：保存数据，并参与后面的模型计算。

---

## 10～15 分钟：nn.Module 管模型

模型写成：

```python
class Line(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        return self.linear(x)
```

`nn.Linear(1, 1, bias=False)` 里面有一个可以训练的参数 `weight`。它做的计算是：

```text
y_pred = weight × x
```

为了和第 1 单元保持一致，代码把初始 weight 设成 `0`：

```python
model = Line()

with torch.no_grad():
    model.linear.weight.zero_()
```

执行：

```python
y_pred = model(x)
```

PyTorch 会自动调用 `forward()`。第一轮的计算是：

```text
y_pred = 0 × 2 = 0
```

`nn.Module` 主要帮我们管理两件事：

1. 模型收到输入以后怎样计算预测。
2. 模型里面有哪些需要训练的参数。

所以后面可以用 `model.parameters()` 一次拿到模型的所有参数，不需要自己维护一份 `w、b、w1、w2...` 列表。

---

## 15～20 分钟：loss.backward() 算梯度

预测值出来以后，先计算 loss：

```python
criterion = nn.MSELoss()
loss = criterion(y_pred, y)
```

第一轮：

```text
y_pred = 0
y = 6
loss = (0 - 6)² = 36
```

接着执行：

```python
loss.backward()
```

它会计算 loss 相对于模型参数的梯度，并写进参数的 `.grad`：

```python
print(model.linear.weight.grad)  # tensor([[-24.]])
```

第 1 单元中，我们自己写梯度公式：

```python
gradient = 2 * error * x
```

代入第一轮的数据：

```text
error = y_pred - y = -6
gradient = 2 × (-6) × 2 = -24
```

PyTorch 的 `backward()` 得到的也是 `-24`。

这里最容易混淆的是 `loss` 和梯度：

```text
loss = error²
梯度 = loss 相对于 weight 的变化率
```

`2 × error` 不是新的 loss。完整关系是：

```text
weight 改变 Δw
→ error 改变 x × Δw
→ loss 大约改变 2 × error × x × Δw
```

所以 loss 相对于 weight 的变化率是：

```text
2 × error × x
```

这就是 `loss.backward()` 自动算出的东西。它只计算梯度，不会修改参数。

---

## 20～25 分钟：optimizer.step() 更新参数

先创建优化器：

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
```

这里告诉优化器两件事：

```text
要更新谁：model.parameters()
每次走多远：lr = 0.05
```

调用：

```python
optimizer.step()
```

它会对模型中的每个参数执行：

```text
parameter = parameter - learning_rate × parameter.grad
```

第一轮就是：

```text
weight = 0 - 0.05 × (-24)
       = 1.2
```

所以两个方法的分工是：

| 代码 | 做什么 | 会不会修改参数 |
|---|---|---|
| `loss.backward()` | 计算梯度，写入参数的 `.grad` | 不会 |
| `optimizer.step()` | 使用 `.grad` 更新参数 | 会 |

循环开始还有一行：

```python
optimizer.zero_grad()
```

PyTorch 默认会累加梯度。这行负责清掉上一轮的梯度，避免新旧梯度混在一起。现在记住它要放在 `backward()` 前面即可。

---

## 25～30 分钟：把四部分串起来

重新看完整训练循环：

```python
for step in range(steps):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
```

逐行翻译：

| 代码 | 这一步发生了什么 |
|---|---|
| `optimizer.zero_grad()` | 清掉上一轮梯度 |
| `model(x)` | `nn.Module` 使用 Tensor 算预测 |
| `criterion(y_pred, y)` | 比较预测和真实答案，得到 loss |
| `loss.backward()` | 自动计算每个参数的梯度 |
| `optimizer.step()` | 使用梯度更新每个参数 |

和第 1 单元对比：

| 第 1 单元手写 | PyTorch |
|---|---|
| 自己保存 `w` | `nn.Module` 管理参数 |
| `predict(w, x)` | `model(x)` |
| 自己写梯度公式 | `loss.backward()` |
| `w -= LR * gradient` | `optimizer.step()` |

PyTorch 自动做了三件事：

1. `nn.Module` 收集并管理模型参数。
2. `loss.backward()` 根据计算过程求出梯度。
3. `optimizer.step()` 更新全部参数。

我们仍然要决定模型结构、loss、优化器、学习率和训练多少次。

### 动手验证

把 [pytorch_regression.py](./pytorch_regression.py) 中的学习率从：

```python
lr=0.05
```

改成：

```python
lr=0.1
```

运行前先计算第一轮更新后的 weight：

```text
weight = 0 - 0.1 × (-24) = 2.4
```

再运行脚本检查输出。验证后把学习率改回 `0.05`。

## 学完检查

不看上面的内容，回答下面 6 个问题：

1. Tensor 在这个例子中装了什么？
2. `nn.Module` 管理哪两类信息？
3. `model(x)` 为什么会执行 `forward()`？
4. `loss.backward()` 把结果写到哪里？
5. `optimizer.step()` 使用什么信息更新参数？
6. `backward()` 和 `step()` 的分工是什么？

能解释下面这条路径，就完成了本单元：

```text
Tensor 提供数据
→ nn.Module 算预测并管理参数
→ loss.backward() 计算参数梯度
→ optimizer.step() 更新参数
```

## 学完以后再看

主线到这里已经完成。下面两个脚本只用于验证和扩展，不属于本次 30 分钟内容：

- [pytorch_regression_full.py](./pytorch_regression_full.py)：把一个点、一个 weight 扩展成 25 个点、weight 和 bias。训练循环仍然是同样的 5 行。
- [handwritten_vs_pytorch.py](./handwritten_vs_pytorch.py)：让手写梯度和 PyTorch 从同一个起点训练，验证两边得到完全相同的参数。

`zero_grad_demo.py` 不再作为主线材料。当前阶段知道梯度会累加，并在每轮 `backward()` 前调用 `optimizer.zero_grad()`，已经足够。
