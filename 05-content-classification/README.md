# 05 完整内容分类项目：训练一个能接收图片的模型

前四章分别练过训练循环、二分类指标和图片分类。这一章不再引入一种全新的训练原理，而是把已经见过的东西串成一个完整项目：

```text
图片数据
→ 训练 ResNet18
→ 用 validation 选择模型和 threshold
→ 用 test 做最终验收
→ 查看 Bad Case
→ 保存 checkpoint
→ 通过 FastAPI 接收一张新图片
```

主线任务先只做两个类别：

```text
normal = 正常
unsafe = 违规
```

学习中遇到的问题可以继续记录在：[问答录](./问答录.md)。

## 学完后要真正会什么

不是记住 ResNet18 的内部结构，而是能独立回答并完成这些事情：

1. 一张图片以什么形式进入模型，模型最终输出什么。
2. 为什么不从零训练图片模型，而是使用迁移学习。
3. train、validation、test 在真实项目中分别在哪里使用。
4. 为什么模型输出概率后，还要单独选择 threshold。
5. 如何区分 FP（误杀）和 FN（漏放），并根据 Bad Case 决定下一步收集什么数据。
6. checkpoint 里保存了什么，FastAPI 怎样使用同一个 checkpoint 做预测。

## 先看最终成品长什么样

训练完成后，模型的主干结构可以先简化成下面这样：

```text
一张 RGB 图片
尺寸调整为 224 × 224
        ↓
ResNet18 已学过的图片特征提取能力
        ↓
512 个图片特征
        ↓
我们新训练的最后一层 Linear(512, 2)
        ↓
[normal_logit, unsafe_logit]
        ↓ softmax
{normal: 0.82, unsafe: 0.18}
        ↓ 和 threshold 比较
最终判断：normal
```

模型不是直接输出一句“这张图违规”。它先输出两个 `logit`，再通过 `softmax` 变成总和为 1 的两个概率。

如果保存的 threshold 是 `0.35`：

```text
unsafe 概率 >= 0.35 → unsafe
unsafe 概率 <  0.35 → normal
```

因此，完整预测包含两部分：

```text
模型参数决定概率
业务 threshold 决定最终动作
```

## 很重要：演示数据不是真实审核数据

本章附带的数据生成器只会画安全的圆形和警告三角形：

```text
圆形图片       → 暂时贴上 normal 标签
警告三角图片   → 暂时贴上 unsafe 标签
```

这里的 `unsafe` 只是一个用于练习流程的名字。模型学到的是“圆形和三角形的区别”，不具备识别色情、暴力或武器的能力。

这么做是为了让你先安全、快速地跑通：

```text
数据目录 → 训练 → 指标 → Bad Case → checkpoint → API
```

真正做内容审核时，最重要的工作之一是先写清审核规则，并收集符合规则、覆盖真实分布且合法合规的标注数据。只替换目录里的图片，不一定就能得到可上线的审核模型。

## 第一次运行：先看到完整结果

下面的命令都在仓库根目录执行。

### 1. 生成演示数据

```bash
uv run python 05-content-classification/generate_demo_data.py
```

生成的目录结构是：

```text
05-content-classification/.data/demo/
├── train/
│   ├── normal/
│   └── unsafe/
├── validation/
│   ├── normal/
│   └── unsafe/
└── test/
    ├── normal/
    └── unsafe/
```

目录名就是标签。读取 `train/unsafe/unsafe_001.png` 时，`ImageFolder` 会同时得到图片和 `unsafe` 标签。

### 2. 训练模型

```bash
uv run python 05-content-classification/train.py
```

第一次运行可能会下载约 45 MB 的 ResNet18 预训练参数。训练结束后会生成：

```text
05-content-classification/artifacts/content_resnet18.pt
```

这是 checkpoint，也就是训练成果。没有它，预测脚本和 API 就没有已经学好的参数可用。

### 3. 让模型预测一张没参与训练的图片

```bash
uv run python 05-content-classification/predict.py \
  05-content-classification/.data/demo/test/unsafe/unsafe_010.png
```

输出类似：

```text
图片：.../unsafe_010.png
normal：16.36%
unsafe：83.64%
threshold：0.40
最终判断：unsafe
```

重点不是这几个具体数字，而是能读懂：模型给了两个概率，程序再用保存好的 threshold 得到最终判断。

### 4. 生成 Bad Case 报告

```bash
uv run python 05-content-classification/analyze_bad_cases.py
```

打开 [Bad Case 报告](./reports/bad_cases.html)，逐个观察模型判断错的图片。

默认分析 `validation`，因为我们还处在理解和改进模型的阶段。不要反复查看 `test` 并据此调参，否则 test 也会逐渐变成另一份 validation。

演示数据较简单，validation 可能没有误判。完成最终 test 验收后，可以额外生成一次 test 错误报告，练习阅读 FP/FN：

```bash
uv run python 05-content-classification/analyze_bad_cases.py \
  --split test \
  --output 05-content-classification/reports/test_bad_cases.html
```

打开 [test 错误报告](./reports/test_bad_cases.html)。它只用于解释最终验收结果，不要再根据其中的样本调整本轮模型。

### 5. 启动 API

```bash
uv run uvicorn api:app \
  --app-dir 05-content-classification \
  --host 127.0.0.1 \
  --port 8005
```

浏览器打开：

- 接口页面：[http://127.0.0.1:8005/docs](http://127.0.0.1:8005/docs)
- 健康检查：[http://127.0.0.1:8005/health](http://127.0.0.1:8005/health)

也可以直接请求：

```bash
curl -X POST http://127.0.0.1:8005/predict \
  -F 'file=@05-content-classification/.data/demo/test/unsafe/unsafe_010.png'
```

## 第一步：先定义模型要回答的问题

“识别违规内容”还不是一个足够明确的任务。模型至少需要知道：

```text
哪些内容算 unsafe？
边界案例算哪一类？
一张图既有多种风险时怎样标？
误杀和漏放哪个代价更高？
```

本章为了专注训练过程，把问题简化为：

> 输入一张图片，输出它属于 `normal` 和 `unsafe` 的概率，并根据 threshold 给出二分类判断。

这是一个图片二分类任务。模型训练时看到的是：

```text
x = 图片像素
y = 人给出的正确标签（normal 或 unsafe）
```

和前四章完全相同的部分是：

```text
x → model(x) → logits → loss → backward → 更新参数
```

只是 `x` 从几个数字变成了一张图片，模型从一两层变成了 ResNet18。

## 第二步：理解迁移学习解决什么问题

如果从随机参数开始训练 ResNet18，它需要大量图片、计算资源和训练时间，才能先学会边缘、纹理、形状等基础视觉特征。

迁移学习的做法是：

```text
拿一个已经在大量图片上训练过的 ResNet18
→ 保留它提取通用图片特征的能力
→ 把原来输出 1000 类的最后一层换掉
→ 只训练新的 normal / unsafe 分类层
```

对应代码在 [model.py](./model.py)：

```python
model = resnet18(weights=ResNet18_Weights.DEFAULT)

for parameter in model.parameters():
    parameter.requires_grad = False

model.fc = nn.Linear(512, 2)
```

可以把它理解成：

```text
冻结的 ResNet18：把图片翻译成 512 个特征
新 fc 层：学习怎样根据这 512 个特征区分 normal / unsafe
```

`requires_grad = False` 表示训练时不更新这些参数。当前项目只更新最后的 `fc` 层，因此训练快，也不容易用少量数据破坏预训练模型原有的视觉能力。

这并不表示 ResNet18 永远不能更新。等数据更多、基线稳定后，可以解冻部分或全部主干，用更小的 learning rate 微调。那是后续优化，不是第一次跑通时必须做的事。

## 第三步：图片进入模型前发生了什么

不同图片可能有不同尺寸和像素范围，不能原样组成一个 batch。预处理代码在 [data.py](./data.py)。

训练图片会经过：

```text
随机裁剪到 224 × 224
→ 随机水平翻转
→ 转成 Tensor
→ 按 ImageNet 的均值和标准差归一化
```

随机裁剪和翻转叫数据增强。它们不会更新模型参数，而是在训练时制造略有不同的输入，减少模型死记某个固定位置的机会。

validation、test 和实际预测不能随机变化，所以使用固定的：

```text
Resize → CenterCrop → Tensor → Normalize
```

这里有一个容易忽略的规则：

> 训练、验证和线上预测的基础归一化必须一致，否则同一张图片会以不同的数值形式进入模型。

## 第四步：训练循环其实没有变

打开 [train.py](./train.py)，核心仍然是前面学过的四行：

```python
logits = model(images)
loss = criterion(logits, labels)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

它们分别在做：

| 代码 | 当前项目里的含义 |
|---|---|
| `model(images)` | 每张图片输出两个 logits |
| `criterion(logits, labels)` | 计算预测和正确类别相差多少 |
| `loss.backward()` | 计算最后一层参数应该怎样变化 |
| `optimizer.step()` | 根据梯度真正修改最后一层参数 |

使用的是 `CrossEntropyLoss`。它适合“每张图只能属于一个类别”的多分类任务，也能用于现在的两个互斥类别。

训练时一个 batch 的形状大致是：

```text
images: [16, 3, 224, 224]
labels: [16]
logits: [16, 2]
```

含义是：一次送入 16 张图片；每张图片有 RGB 三个通道；模型为每张图片输出两个类别分数。

## 第五步：validation 不只用来看分数

每个 epoch 结束后，代码会在 validation 上完成两件事：

```text
1. 计算当前模型的 Precision、Recall、F1
2. 在 0.10～0.90 中选择 validation F1 最好的 threshold
```

然后只保存 validation F1 最好的 checkpoint：

```python
if validation_metrics["f1"] > best_f1:
    save_checkpoint(...)
```

为什么不直接保存最后一个 epoch？

```text
训练继续进行
→ train 表现可能继续变好
→ validation 表现却可能开始下降
```

这就是 03 学过的过拟合。保存 validation 最好的模型，相当于保留模拟考试表现最好的那一次，而不是盲目保留做题最久的那一次。

## 第六步：test 为什么只在最后出现

模型、epoch 和 threshold 都根据 validation 选完后，代码才加载最佳 checkpoint，并在 test 上评估一次：

```text
train：更新模型参数
validation：选择 checkpoint 和 threshold
test：最终验收
```

如果看到 test 结果不好就回去改参数，再回来查看 test，test 已经影响了你的决策。这样反复几次后，test 就不再代表真正未知的数据。

真实项目中，修改方案后应该建立新的、未参与决策的验收集，或严格控制 test 的使用过程。

## 第七步：Bad Case 比一个 F1 更能告诉你下一步

F1 告诉你“整体有多好”，但不会告诉你“为什么错”。[analyze_bad_cases.py](./analyze_bad_cases.py) 会把错误分成两类：

| 类型 | 真实标签 | 模型判断 | 内容审核含义 |
|---|---|---|---|
| FP | normal | unsafe | 误杀了正常内容 |
| FN | unsafe | normal | 漏放了违规内容 |

看 Bad Case 时不要只说“模型不准”，而要找成组的原因：

```text
某种背景颜色总被误判？
小目标、遮挡或裁剪后容易漏掉？
训练集中某一类场景几乎没有？
标签规则本身是否不一致？
阈值过低导致 FP 多，还是模型概率本身就错了？
```

接下来采取的动作也不同：

```text
数据覆盖不足 → 补同类真实样本
标签不一致   → 明确规则并重新标注
FP 太多      → 检查 normal 困难样本和 threshold
FN 太多      → 补 unsafe 困难样本，或降低 threshold
模型能力不足 → 再考虑解冻主干、换模型或增加输入信息
```

这就是完整项目里的一个重要闭环：

```text
训练 → 看指标 → 看 Bad Case → 找错误模式 → 改数据或方案 → 重新训练
```

## 第八步：checkpoint 保存了什么

[model.py](./model.py) 保存两类信息：

```text
model_state_dict：模型学到的参数
metadata：类别映射、threshold、最佳 epoch、validation 指标等
```

只保存模型参数、不保存类别顺序会有风险。例如训练时：

```text
0 = normal
1 = unsafe
```

线上代码却按相反顺序解释，概率就会被读反。threshold 也属于模型交付的一部分，不能只存在训练者的脑子里。

## 第九步：FastAPI 没有重新训练模型

[api.py](./api.py) 启动时做的是：

```text
读取 checkpoint
→ 重建同样的 ResNet18 结构
→ 加载训练好的参数和 metadata
→ 等待图片上传
```

每次 `/predict` 请求只执行：

```text
读取图片 → 同样的预处理 → forward → softmax → threshold → JSON
```

它不会执行 `loss.backward()` 和 `optimizer.step()`，所以这是推理，不是训练。

API 返回示例：

```json
{
  "filename": "sample.png",
  "decision": "unsafe",
  "threshold": 0.5,
  "probabilities": {
    "normal": 0.08,
    "unsafe": 0.92
  }
}
```

这一章的学习重点不是 FastAPI 的写法，而是确认 API 使用的模型、图片预处理、类别映射和 threshold 与训练阶段完全一致。

## 建议的代码阅读顺序

不要按文件名逐个硬啃，按数据流阅读：

1. [generate_demo_data.py](./generate_demo_data.py)：标签和目录怎样形成。
2. [data.py](./data.py)：图片怎样变成模型输入。
3. [model.py](./model.py)：预训练 ResNet18 怎样被改成两个输出。
4. [train.py](./train.py)：训练、validation、保存最佳 checkpoint。
5. [inference.py](./inference.py)：一张新图片怎样变成最终判断。
6. [analyze_bad_cases.py](./analyze_bad_cases.py)：怎样找到 FP 和 FN。
7. [api.py](./api.py)：怎样把同一套推理暴露为接口。

## 换成自己的数据

先准备完全相同的目录结构：

```text
my-data/
├── train/normal/
├── train/unsafe/
├── validation/normal/
├── validation/unsafe/
├── test/normal/
└── test/unsafe/
```

然后运行：

```bash
uv run python 05-content-classification/train.py --data my-data
```

划分数据前需要防止数据泄漏。例如同一视频相邻帧、同一原图的裁剪版本、同一用户连续上传的近重复图，不应随机散落在 train 和 test 中。否则模型可能只是认出了几乎相同的图片，test 指标会虚高。

对于真实内容数据，还必须考虑隐私、授权、访问控制、标注人员保护和数据留存规则。本章的安全合成数据无法替代这些工作。

## 关于 sexual / violence / weapon / normal

这四个名字看起来像四分类，但真实图片可能同时包含 `sexual` 和 `violence`。如果硬要求四选一，会丢失信息。

更常见的建模方式是多标签分类：

```text
sexual  概率：0.82
violence 概率：0.71
weapon  概率：0.09
```

三个风险标签各自有一个 sigmoid 和 threshold；三个风险都没有触发时，再视为 normal。它和本章二分类的核心训练循环不变，但标签、loss、指标和阈值都会从“一套”变成“每个风险类型一套”。

先把 `normal / unsafe` 主线真正跑懂，再扩展多标签。不要只是把最后一层从 2 改成 4，就认为完成了多风险审核。

## ResNet 和 ViT 现在选哪个

这章先用 ResNet18，因为模型较小、训练快，卷积和迁移学习的路径也更容易观察。

ViT 是另一种图片模型结构，可以作为后续对照实验：保持相同数据划分和指标，只替换模型，然后比较 validation/test 和 Bad Case。第一次学习完整链路时，同时更换数据、模型和训练方法，会很难判断结果为什么变化。

## 学完检查

先不看答案，试着回答：

1. 这个项目的模型输入、logits、概率和最终判断分别是什么？
2. 为什么最后一层是 `Linear(512, 2)`？
3. 冻结 ResNet18 主干以后，训练时哪些参数会变化？
4. 为什么训练图片可以随机裁剪，而 validation 和预测图片不能随机变化？
5. validation 在这里做了哪两件事？
6. 为什么 test 不能用来反复选择 threshold？
7. FP 和 FN 各对应哪一种审核错误？
8. checkpoint 除了模型参数，为什么还要保存类别映射和 threshold？
9. FastAPI 收到图片后有没有训练模型？它实际做了什么？
10. 为什么 `sexual / violence / weapon / normal` 可能更适合多标签，而不是四选一？

参考答案在：[问答录](./问答录.md)。

## 参考资料

- [PyTorch 迁移学习教程](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Torchvision ResNet18 文档](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)
- [PyTorch 保存和加载模型](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
