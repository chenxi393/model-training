# Logistic Regression：第一个二分类模型

预计学习时间：30 分钟。

这次要解决的问题是：给一条数据，判断它属于 `0` 还是 `1`。

例如内容审核可以把标签定义为：

```text
0 = normal
1 = unsafe
```

学完后，你应该能解释：

- train、validation、test 为什么要分开
- 模型怎样把输入变成 0～1 的概率
- Accuracy、Precision、Recall、F1 各自回答什么问题
- 为什么训练集表现好，不代表模型真的好

配套代码：[binary_classification.py](./binary_classification.py)

学习中遇到的问题统一记录在：[问答录](./问答录.md)

---

## 0～5 分钟：先运行

在仓库根目录执行：

```bash
uv run python 03-logistic-regression/binary_classification.py
```

输出分成四部分：

```text
先看一个 Accuracy 陷阱
数据划分
训练过程
最终评估
```

第一部分会出现一个看起来不错、实际完全没用的结果：

```text
Accuracy=0.900  Precision=0.000  Recall=0.000  F1=0.000
```

100 条数据里只有 10 条正类。模型把所有数据都猜成负类，猜对了 90 条，所以 Accuracy 是 90%。但它一条正类都没找到，Recall 和 F1 都是 0。

这就是为什么二分类不能只看 Accuracy。

---

## 5～10 分钟：Logistic Regression 在算什么

线性回归输出一个任意大小的数字：

```text
z = w₁x₁ + w₂x₂ + b
```

这个数字叫 `logit`。它可能是 `-3.2`，也可能是 `5.7`，还不能直接当概率。

Logistic Regression 会再经过一个 `sigmoid`：

```text
logit z
   ↓ sigmoid
0～1 之间的概率
```

几个例子：

| logit | sigmoid 后的概率 | 使用 0.5 阈值 |
|---:|---:|---|
| `-2` | 约 `0.12` | 预测为 0 |
| `0` | `0.50` | 预测为 1 |
| `2` | 约 `0.88` | 预测为 1 |

代码中的模型只有一层：

```python
model = nn.Linear(2, 1)
```

它接收两个特征，输出一个 logit。预测时再转换成概率和标签：

```python
logits = model(x)
probabilities = torch.sigmoid(logits)
labels = (probabilities >= 0.5).int()
```

训练时使用：

```python
criterion = nn.BCEWithLogitsLoss()
```

它把 sigmoid 和二分类交叉熵合在了一起，直接接收 logits。当前只需要知道：预测越偏离真实标签，loss 越大。

---

## 10～15 分钟：为什么要分三份数据

脚本生成 1000 条数据，然后只划分一次：

```text
train       600 条
validation  200 条
test        200 条
```

三份数据的职责不同：

| 数据 | 用来做什么 | 模型训练时能不能看 |
|---|---|---|
| train | 计算 loss、反向传播、更新参数 | 能 |
| validation | 选择阈值、比较方案 | 能评估，不能更新参数 |
| test | 模拟模型遇到真正没见过的数据 | 最后才看一次 |

训练集相当于平时做过的练习题。模型反复看这些数据，训练集分数变高是正常的。

validation 相当于模拟考试，用来发现模型是不是只记住了训练数据。

test 相当于最后考试。如果一边看 test 分数一边改模型，test 就参与了调参，不能再代表“没见过的数据”。

所以真正关心的是：模型在 validation 和 test 上是否也表现良好。

---

## 15～20 分钟：训练循环没有变

打开 [binary_classification.py](./binary_classification.py)，找到训练循环：

```python
logits = model(x_train)
loss = criterion(logits, y_train)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

它和第 2 单元还是同一件事：

```text
输入 train 数据
→ 算预测
→ 算 loss
→ backward 算梯度
→ optimizer 更新参数
```

注意训练循环只把 `x_train` 和 `y_train` 交给模型。validation 和 test 都不参与参数更新。

脚本每隔一段时间同时打印 train F1 和 validation F1：

```text
epoch=...  loss=...  train_F1=...  validation_F1=...
```

如果 train F1 持续上升，而 validation F1 开始下降，说明模型越来越会做训练题，但对新数据反而变差。这种现象叫 overfitting，也就是过拟合。

这就是“训练集表现好，不代表模型真的好”的具体含义。

---

## 20～25 分钟：四个指标分别看什么

先认识混淆矩阵中的四种结果：

| 名称 | 含义 | 内容审核示例 |
|---|---|---|
| TP | 正类预测成正类 | unsafe 被拦截 |
| TN | 负类预测成负类 | normal 被放行 |
| FP | 负类误判成正类 | normal 被误杀 |
| FN | 正类误判成负类 | unsafe 被漏放 |

四个指标都从这四个数算出来。

### Accuracy：总体猜对多少

```text
Accuracy = (TP + TN) / 全部数据
```

类别比较均衡时很直观。类别很不均衡时，它可能骗人。前面“全猜负类”的模型就是例子。

### Precision：判为正类的结果有多可信

```text
Precision = TP / (TP + FP)
```

Precision 低，说明误杀多。模型拦下来的内容里，有很多其实是 normal。

### Recall：真正的正类找回了多少

```text
Recall = TP / (TP + FN)
```

Recall 低，说明漏放多。真实的 unsafe 内容没有被模型发现。

### F1：同时考虑 Precision 和 Recall

```text
F1 = 2 × Precision × Recall / (Precision + Recall)
```

只有 Precision 和 Recall 都不错，F1 才会高。需要用一个数字比较二分类模型时，F1 通常比单独看 Accuracy 更可靠。

---

## 25～30 分钟：改变 threshold

概率必须经过 threshold 才能变成 0 或 1：

```python
labels = (probabilities >= threshold).int()
```

脚本在 validation 上比较三个阈值：

```text
threshold = 0.3
threshold = 0.5
threshold = 0.7
```

一般来说：

```text
threshold 降低 → 更容易判为正类 → Recall 往往上升，FP 也可能增加
threshold 提高 → 更难判为正类 → Precision 可能上升，FN 也可能增加
```

脚本选择 validation F1 最高的阈值，然后只在最后使用 test 评估一次。

这条顺序很重要：

```text
train 训练参数
validation 选择方案和 threshold
test 只负责最终验收
```

## 学完检查

不看上面的内容，回答下面 6 个问题：

1. Logistic Regression 输出的 logit 和概率有什么区别？
2. train、validation、test 各自负责什么？
3. 为什么不能根据 test 结果反复调整 threshold？
4. Accuracy 很高时，模型为什么仍然可能没用？
5. FP 和 FN 在内容审核中分别代表什么？
6. 降低 threshold 通常会怎样影响 Recall？

## 参考答案

先自己回答上面的 6 个问题，再对照这里：

1. **Logit 和概率**：logit 是模型输出的原始分数，范围没有限制。概率由 `sigmoid(logit)` 得到，范围在 `0` 到 `1` 之间。
2. **三份数据**：train 用来更新模型参数；validation 用来选择模型方案和 threshold；test 在所有选择完成后只评估一次。
3. **不能反复调 test**：一旦根据 test 结果调整 threshold，test 就参与了调参，不能再代表公平的最终评估。如果已经用 test 调过很多次，需要重新准备一份没看过的数据作为最终 test。
4. **Accuracy 可能骗人**：类别不均衡时，模型全部预测为多数类，也可能得到很高的 Accuracy，但 Recall 和 F1 可能为 `0`。
5. **FP 和 FN**：FP 是实际正常却被预测为违规，也就是误杀；FN 是实际违规却被预测为正常，也就是漏放。
6. **降低 threshold**：模型更容易预测为正类，Recall 通常会上升，FN 通常减少，但 FP 可能增加，Precision 可能下降。

能回答这些问题，并能读懂下面这条完整路径，就完成了本单元：

```text
特征 x
→ Linear 输出 logit
→ sigmoid 得到概率
→ threshold 得到 0/1 标签
→ TP / TN / FP / FN
→ Accuracy / Precision / Recall / F1
```

## 课后练习

下面的练习不计入本次 30 分钟：

1. 把候选阈值改成 `0.2、0.4、0.6、0.8`，观察 Precision 和 Recall。
2. 把训练轮数从 `200` 改成 `20`，比较 train 和 validation 的 F1。
3. 修改 `classification_metrics()`，只保留 TP、FP、FN、TN，自己重新写四个指标公式。
