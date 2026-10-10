# 06 训练一个小型 Transformer：根据前文预测下一个字

本节的目标是：

> 训练一个字符语言模型，输入前面的文字，输出下一个字符的概率；能解释输入怎样变成向量、模型怎样读取前文、loss 怎样让参数发生变化。

沿用 01–05 的训练循环：

```text
输入 → 预测 → 和正确答案比较得到 loss → backward → optimizer.step
```

新增的内容是处理文本和上下文的方法。你不需要先把全部数学补完，也不用以“独立默写 Transformer”为过关标准。先能讲清下面的计算、完成训练和预测，再尝试自己改写核心代码。

## 学习路线与过关标准

每走完一步，先回答这一小步的检查题，再继续。能运行脚本只是开始；能解释一个具体结果，才说明理解得更扎实。

| 顺序 | 要解决的问题 | 对应材料 |
| --- | --- | --- |
| 0 | 模型到底做哪一道题？ | [next_token_demo.py](./next_token_demo.py) |
| 1 | 字符怎样变成数字？特征和分数有什么区别？ | [embedding_demo.py](./embedding_demo.py)、[position_embedding_demo.py](./position_embedding_demo.py) |
| 2 | 为什么最后一个字需要读取前文？ | [attention_purpose_demo.py](./attention_purpose_demo.py) |
| 3–5 | 点积、softmax、加权求和分别怎样计算？ | [attention_math_demo.py](./attention_math_demo.py) |
| 6 | Q/K/V 怎样来？怎样遮住未来？ | [Attention 实验台](./attention-lab.html) |
| 7 | 训练前后的预测概率、loss 和参数怎样变化？ | [prediction_learning_demo.py](./prediction_learning_demo.py) |
| 8 | 怎样组合成 Transformer Block？ | [model.py](./model.py) |
| 9 | 怎样训练完整语言模型并生成文字？ | [train.py](./train.py)、[generate.py](./generate.py) |

所有命令在仓库根目录执行。具体问题可以记入 [问答录](./问答录.md)。

## 0. 先看模型在做哪一道题

已有文本：

```text
我吃了饭。
```

从这句话里可以直接取出题目和答案：

```text
看到“我”     → 正确的下一个字符是“吃”
看到“我吃”   → 正确的下一个字符是“了”
看到“我吃了” → 正确的下一个字符是“饭”
```

答案来自训练文本。现实中“我吃了”后面也可能是“面”，这里只是在学习这条具体样本。看到足够多样本后，模型才能学习不同后续出现的概率。

运行：

```bash
uv run python 06-tiny-transformer/next_token_demo.py
```

脚本用“模型学习。”演示同一件事：截取前缀作输入，右边紧挨着的字符作答案。这里还没有训练，只是在准备题目。

**检查：** 原文是“天黑了。”，输入“天黑了”时，正确答案是什么？

<details>
<summary>核对答案</summary>

是“。”。训练时必须根据原文取答案，不能让程序随便猜一个答案当标签。

</details>

## 1. Embedding：字符表示和输出分数要分开

先回顾你已经见过的过程：

```text
“模” → ID 4 → 查 Embedding 表第 4 行 → [0.2, -0.5, 0.8]
```

上面的 ID 和小数都是举例。ID 是查表行号；这一行的小数是模型计算使用的特征表示。向量就是按顺序排好的一组数。

```bash
uv run python 06-tiny-transformer/embedding_demo.py
```

Embedding 表创建时，小数随机初始化。训练时 loss 的梯度传回这张表，优化器修改这些小数。字符 ID 的映射保持不变。

### 这三个小数是不是模型给“模”的分数？

它们是输入特征。它们还要经过后面的网络，最终才会得到各个候选答案的分数。

| 名称 | 回答什么问题 | 举例 |
| --- | --- | --- |
| token ID | 去表里哪一行找这个字符？ | 4 |
| Embedding 向量 | 这个输入字符用哪组数参与计算？ | [0.2, -0.5, 0.8] |
| logits | 各个候选答案得多少分？ | “饭”得 2.1，“。”得 -0.3 |
| softmax 概率 | 这些候选答案各占多少概率？ | “饭”占 0.65，“。”占 0.06，其他字符占剩余部分 |

这里的数均为说明概念的示例。Embedding 的每一维通常没有固定的人类含义，不能直接解释为“第一维是懂了多少”“第二维是违规程度”。

你在 05 见过同样的区别：

```text
ResNet 的 512 个特征 → 最后一层 → normal / unsafe 的两个 logit
字符的 Embedding    → 后面的网络 → 每个候选字符的 logit
```

### Position Embedding 补充位置信息

同一个“模”无论出现在哪里，查到的 Token Embedding 都相同。再给位置编号查另一张表，就能补充位置：

```text
当前位置初始向量 x = 字符向量 + 位置向量
```

运行：

```bash
uv run python 06-tiny-transformer/position_embedding_demo.py
```

**检查：** 训练之后，字符 ID、Embedding 小数、输出 logits，哪些可能改变？Embedding 的第一维是不是固定代表一种含义？

<details>
<summary>核对答案</summary>

ID 映射不变，Embedding 小数会被优化器更新；同样输入的 logits 也可能随着模型参数更新而改变。Embedding 各维通常没有预先指定的语义。

</details>

## 2. 为什么必须读取前文

我们准备两道题：

```text
“我吃了” → “饭”
“天黑了” → “。”
```

两个输入最后都是“了”，位置也都是 2（位置从 0 开始）。所以在读取上下文之前，最后位置的初始向量相同：

```text
“了”的 Token Embedding + 位置 2 的 Position Embedding
```

如果后续只对这一个向量计算，两道题就会得到相同的输出。要学会给出不同答案，当前位置需要从“我、吃”或“天、黑”读取信息。

```bash
uv run python 06-tiny-transformer/attention_purpose_demo.py
```

Attention 提供一种读取机制：计算对每个可见位置的关注权重，再按权重混合它们的信息。它是否学会有用的读取方式，取决于参数和训练。

**检查：** Position Embedding 已经告诉模型“了”在位置 2，为什么还不够？

<details>
<summary>核对答案</summary>

两道题中的字符和位置都相同，但前文不同。位置向量本身不包含其他位置的内容，还需要读取前文。

</details>

## 3. 点积：两组数怎样算出一个匹配分数

先练习最基本的运算。下面两组数是人为选的，方便手算：

```text
a = [1, 2]
b = [3, 4]
```

点积就是对应位置相乘，再加起来：

```text
a · b = 1×3 + 2×4 = 11
```

在 Attention 中，一组数来自当前位置，另一组来自被查询的位置。点积把这两组数变成一个匹配分数：

```text
当前查询向量 [1, 2]

和 [1, 0] 匹配：1×1 + 2×0 = 1
和 [0, 1] 匹配：1×0 + 2×1 = 2
和 [1, 1] 匹配：1×1 + 2×1 = 3
```

这时只有分数 [1, 2, 3]，还不是概率。点积受方向和大小共同影响，不是天然可靠的“语义相似度”；模型要通过训练让这种匹配有助于预测。

运行下面的脚本，它会依次打印本节 3–5 步：

```bash
uv run python 06-tiny-transformer/attention_math_demo.py
```

**检查：** [2, 1] 和 [3, 4] 的点积是多少？

<details>
<summary>核对答案</summary>

2×3 + 1×4 = 10。

</details>

## 4. softmax：把多个分数变成一组权重

我们有三个分数：

```text
[1, 2, 3]
```

想用它们决定读取比例，需要得到非负、总和为 1 的权重。softmax 的计算是：

```text
先取指数：exp(1), exp(2), exp(3)
         ≈ 2.718, 7.389, 20.086
加起来： ≈ 30.193

再分别除以总和：
[2.718/30.193, 7.389/30.193, 20.086/30.193]
≈ [0.090, 0.245, 0.665]
```

exp(s) 就是 e 的 s 次方，e 约为 2.718。此处先把它当成“把任意分数变成正数，并保留高低顺序”的函数即可。

三个权重的意思是读取比例：

```text
第一个位置约占 9.0%
第二个位置约占 24.5%
第三个位置约占 66.5%
```

这里的 softmax 在位置之间分配关注程度；05 的 softmax 在类别之间分配概率。同一个运算，输入对象和用途不同。

**检查：** 分数都相同，例如 [2, 2, 2]，权重会怎样？这些权重是不是下一个字符的概率？

<details>
<summary>核对答案</summary>

每个权重都是 1/3。这里是读取各个位置的关注权重，后面还需要网络和输出层，才会得到候选字符的概率。

</details>

## 5. 加权求和：按比例读取信息

上一小步得到读取比例。每个位置还有一组可供读取的数字，先用下面的人为示例：

```text
V1 = [2, 0]
V2 = [0, 2]
V3 = [1, 1]
```

每组数字乘自己的权重，再相加：

```text
0.090 × [2, 0] = [0.180, 0.000]
0.245 × [0, 2] = [0.000, 0.490]
0.665 × [1, 1] = [0.665, 0.665]

相加 ≈ [0.845, 1.155]
```

这个新向量混入了多个位置的信息。它仍然是供后续计算的特征，不能当成两个候选答案的概率。

**检查：** 如果权重是 [1, 0, 0]，结果是什么？如果是 [0.5, 0.5, 0] 呢？

<details>
<summary>核对答案</summary>

分别是 [2, 0] 和 [1, 1]。前者只读取第一个位置，后者平分读取前两个位置。

</details>

## 6. Q/K/V：匹配用的数和读取用的数怎样来

你已经知道三件事：两组数算匹配分数，softmax 算权重，权重乘信息向量再求和。现在给这几种向量命名：

| 名称 | 来自哪里 | 用来做什么 |
| --- | --- | --- |
| Query / Q | 当前查询位置的 x 经过线性变换 | 和每个 K 做匹配 |
| Key / K | 每个可见位置的 x 经过线性变换 | 提供匹配用的向量 |
| Value / V | 每个可见位置的 x 经过线性变换 | 提供实际被加权读取的信息 |

每个位置都计算 Q、K、V。我们观察某一个位置时，拿它的 Q 查询其他位置的 K，再汇总对应的 V。

### 为什么不直接拿原来的 x 算到底？

分别设置线性变换，让模型能学习“怎样匹配”和“怎样传递内容”这两种不同的计算。三组权重都是训练参数。

先看一个手工设置的二维例子。把线性变换展开成乘法与加法：

```text
x = [1, 2]

Q = [1×1 + 2×0, 1×0 + 2×1] = [1, 2]
K = [1×1 + 2×0, 1×0 + 2×2] = [1, 4]
V = [1×1 + 2×0, 1×1 + 2×1] = [1, 3]
```

这里的乘数是为了演示而人为设置的矩阵元素。真实模型随机初始化这些权重，然后利用 loss 学习。Q、K、V 的每一维也没有固定的人类含义。

打开 [Attention 交互实验台](./attention-lab.html)，从第一步开始：

1. 在“我吃了饭”和“天黑了。”之间切换，保持选中位置 2 的“了”。
2. 看初始向量相同，但上下文不同。
3. 逐步观察点积、softmax 和加权求和。
4. 到 Q/K/V 步骤核对三个不同矩阵怎样产生三种向量。
5. 点击不同位置，检查哪些未来位置会被遮住。

### causal mask：训练时也不能看答案

训练时，整段输入会同时进入模型：

```text
input  = 我 吃 了 饭
target = 吃 了 饭 。
```

位置 2 的“了”要预测“饭”，不能直接读取位置 3 的“饭”。因此每个位置只允许读取自己和左边：

```text
位置 0：可以读 0
位置 1：可以读 0、1
位置 2：可以读 0、1、2
位置 3：可以读 0、1、2、3
```

mask 把未来位置的分数设成负无穷，softmax 后它们的权重就是 0。这样训练时和实际生成时的可见条件一致。

完整计算现在可以串起来：

```text
x → Q/K/V
Q 和各个 K 做点积 → 除以 √head_size → 遮住未来
→ softmax 得到读取权重 → 对 V 加权求和
```

除以 √head_size 是为了控制分数尺度，避免 softmax 过早过于集中。其推导先放到以后。

**检查：** Q/K 影响哪一部分？V 影响哪一部分？不遮住未来会怎样？

<details>
<summary>核对答案</summary>

Q/K 决定匹配分数，进而影响读取权重；V 提供被加权汇总的内容。不遮住未来会让模型在训练时偷看下一个字符，训练效果无法代表实际生成能力。

</details>

## 7. 用真实模型看一次预测、loss 和参数变化

现在先把模型整体当成一个函数，运行：

```bash
uv run python 06-tiny-transformer/prediction_learning_demo.py
```

这个脚本使用本章真正的 TinyTransformer，在 CPU 上训练两道题：

```text
我吃了 → 饭
天黑了 → 。
```

它会打印训练前和训练 150 步后：

```text
每道题的正确答案概率
概率最高的三个候选字符
每道题的 loss
“了”的 Embedding 前三个小数
Q/K/V 权重的变化量
```

这是观察模型记住两道题的实验，没有 validation，也不能证明它能理解其他句子。完整语言模型训练在后面。

本机一次实测结果（具体数值可能因运行环境略有不同）：

| 题目 | 正确概率：训练前 → 训练后 | loss：训练前 → 训练后 |
| --- | --- | --- |
| 我吃了 → 饭 | 0.0426 → 0.9998 | 3.1568 → 0.0002 |
| 天黑了 → 。 | 0.1705 → 0.9998 | 1.7691 → 0.0002 |

两个输入最后的“了”仍使用同一行 Token Embedding，却能给出不同答案。这种区分来自模型对前文的计算；不能把“了”的某一维直接当作“饭”的概率。

### 从特征到 logits，再到概率

对于“我吃了”，模型最后位置会产生一组特征。输出层为词表中的每个字符算一个分数：

```text
最后位置的特征 → Linear → 每个候选字符的 logit → softmax → 候选概率
```

和 05 的区别在输出类别数量：05 输出两个类别的分数，这里输出词表中每个字符的分数。

Attention 的关注权重和最终字符概率是两次不同的 softmax，不能混在一起。

### Cross Entropy 在惩罚什么

这道题的正确答案是“饭”。对于一个确定的正确标签：

```text
loss = -ln(模型给“饭”的概率)
```

ln 是自然对数。先观察对应关系：

| 正确答案的概率 | loss 约为 |
| --- | --- |
| 0.10 | 2.303 |
| 0.50 | 0.693 |
| 0.90 | 0.105 |

模型把概率更多分给正确字符时，loss 下降。训练时 PyTorch 的 cross_entropy 直接接收 logits 和正确字符 ID，内部完成稳定计算：

```python
loss = F.cross_entropy(last_logits, targets)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

梯度会传回输出层、Attention、Embedding 等参与计算的参数。一次训练后正确概率不保证每次都增加，但训练目标是降低整批样本的平均 loss。

**检查：** 正确答案的概率从 0.1 变到 0.9，loss 应该怎样变化？训练时是否只修改最后一层？

<details>
<summary>核对答案</summary>

loss 从约 2.303 降到约 0.105。本例所有可训练参数都参与更新，包括 Embedding、Attention 和输出层；与 05 冻结主干只训练最后一层的设置不同。

</details>

## 8. Transformer Block：把前面的计算组合起来

已经见过 Attention，现在补它周围的组件。本章一个 Block 的代码是：

```python
attention_output, weights = self.attention(self.attention_norm(x))
x = x + attention_output
x = x + self.feed_forward(self.feed_forward_norm(x))
```

| 组件 | 对当前位置做什么 |
| --- | --- |
| Attention | 从可见的其他位置读取信息 |
| Feed Forward / MLP | 用同一套小型神经网络，分别加工每个位置的向量 |
| Residual / 残差连接 | 把新结果加回原来的向量，保留信息并帮助梯度传播 |
| LayerNorm | 对当前位置的特征做归一化，再用可学习参数调整，帮助稳定训练 |

残差相加是逐个数相加，例如：

```text
原向量 [1, 2] + 新结果 [0.3, -0.2] = [1.3, 1.8]
```

Feed Forward 和 04 的 MLP 类似：Linear → 激活函数 → Linear。本章使用 GELU 激活函数，具体曲线可以后看。它不会直接读取其他位置；跨位置交换信息发生在 Attention。

一个 Block 进行一次“读取上下文，再加工”。多个 Block 让这个过程重复进行。

### 多头 Attention 先理解到这里

完整模型让多个 head 分别计算匹配与读取，再拼接并投影输出：

```text
d_model = 64，num_heads = 4 → 每个 head 使用 16 维
```

各个 head 的模式由训练形成，没有预先规定某个 head 负责主语或标点。第一遍只要求知道“多套读取并行计算，再合并”，后续再研究 reshape 和 transpose。

**检查：** Attention 和 Feed Forward 中，哪个会在不同位置之间交换信息？

<details>
<summary>核对答案</summary>

Attention。Feed Forward 对每个位置独立计算，所有位置共享同一套网络参数。

</details>

## 9. 训练完整字符语言模型，再逐字生成

两题实验只监督每段输入的最后位置。完整语言模型会监督所有位置：

```text
input  = 我 吃 了 饭
target = 吃 了 饭 。
```

causal mask 保证每个位置只看自己的前缀，所以可以并行完成多道预测题。

运行：

```bash
uv run python 06-tiny-transformer/train.py
```

默认是两层 Block、64 维特征、4 个 head、48 个字符的上下文，训练 800 步。程序打印实际使用的 device：优先 CUDA，其次 Apple MPS，最后 CPU。

语料前 90% 用来训练，后 10% 用作 validation；这只是教学用的简单切分，真实任务还要考虑文档独立性、重复内容和分布。本章用 validation loss 选择 checkpoint，不另设最终 test。

观察：

```text
train loss 和 validation loss 一起下降 → 对这两份数据的预测都在改善
train loss 下降，validation loss 上升 → 出现过拟合迹象
```

此前默认配置的一次实测：

```text
step=400 train_loss=2.1862 validation_loss=4.5044
step=800 train_loss=0.7335 validation_loss=5.1583
```

那次保存的是 validation 最好的第 400 步。具体数值会随设备和运行有所变化。

checkpoint 保存到：

```text
06-tiny-transformer/artifacts/tiny_transformer.pt
```

里面有模型参数、模型结构配置、字符词表和训练信息。词表解释 ID 的含义，必须随模型保存。

### 生成为什么逐字进行

```bash
uv run python 06-tiny-transformer/generate.py --prompt 模型 --max-new-tokens 80
```

生成的每一轮：

```text
输入当前文本 → 取最后位置的 logits → softmax → 抽取一个字符
→ 追加到文本 → 再预测一次
```

训练时整段答案已经存在，可以并行监督；生成时下一个字符还不存在，需要先生成它才能进入下一轮。

默认模型与语料都很小，生成不通顺很常见。除了观察文本，还要看预测概率、validation loss 和重复情况，不能凭一段输出就判断是否学会原理。

### temperature 是生成设置

```bash
uv run python 06-tiny-transformer/generate.py --prompt 模型 --temperature 0.4
uv run python 06-tiny-transformer/generate.py --prompt 模型 --temperature 1.2
```

在最终 softmax 前把 logits 除以 temperature。较低值让分布更集中，较高值让分布更平坦。它改变采样方式，不修改模型参数，也不会补回模型没学到的知识。

提示词中的字符必须出现在已保存词表里。模型最多读取最近 48 个字符，超出后会截取最近的上下文。

## 现在再看形状和完整代码

最后运行：

```bash
uv run python 06-tiny-transformer/inspect_forward.py
```

这时再用名称定位形状：

```text
input_ids：[B, T]     一批 B 段文本，每段 T 个字符 ID
内部特征：[B, T, C]   每个位置用 C 个数表示
logits：  [B, T, V]   每个位置给词表的 V 个字符分别打分
attention：[B, H, T, T] 每个 head 中，哪个位置读取哪个位置
```

代码阅读顺序：

1. [data.py](./data.py)：input 与 target 错开一位。
2. [model.py](./model.py) 的 TinyTransformer.forward：整体路线。
3. CausalSelfAttention.forward：Q/K/V → 分数 → mask → softmax → 加权求和。
4. TransformerBlock.forward：读取、加工和残差相加。
5. [train.py](./train.py) 与 [generate.py](./generate.py)：参数更新与逐字生成。

## 本节完成标准

完成以下事情，就可以认为主线学会了：

- 从一句文本写出一组 input 和 target。
- 用自己的话区分 ID、Embedding、logits、最终概率。
- 手算一个二维点积和简单的加权求和，说明 softmax 的用途。
- 解释 Q/K/V 的来源，以及 mask 为什么防止偷看答案。
- 运行两题实验，指出哪些参数改变、正确概率和 loss 如何变化。
- 运行完整训练，保存并加载 checkpoint，生成一段文本。
- 说明 Attention 和 MLP 的分工，以及训练与生成的区别。

遇到答不出的点，就回到对应的小实验。先不用展开 BPE、RoPE、KV Cache、Flash Attention 或分布式训练；下一阶段再在这个骨架上学习。

更详细的概念解释放在 [问答录](./问答录.md)。你也可以先口头回答每一步检查题，再对照折叠的参考答案。
