# -*- coding: utf-8 -*-
"""
Logistic Regression 二分类最小示例。

配套讲义：README.md

运行：
    uv run python 03-logistic-regression/binary_classification.py
"""

import torch
import torch.nn as nn


def classification_metrics(y_true, y_pred):
    """根据 0/1 标签计算混淆矩阵和四个常用指标。"""
    y_true = y_true.int().view(-1)
    y_pred = y_pred.int().view(-1)

    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    tn = ((y_true == 0) & (y_pred == 0)).sum().item()
    fp = ((y_true == 0) & (y_pred == 1)).sum().item()
    fn = ((y_true == 1) & (y_pred == 0)).sum().item()

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def predict(model, x, threshold=0.5):
    """把模型输出转换为概率，再按阈值转换为 0/1。"""
    with torch.no_grad():
        logits = model(x)
        probabilities = torch.sigmoid(logits)
        labels = (probabilities >= threshold).int()
    return labels


def show_metrics(name, result):
    print(
        f"{name:<12} "
        f"Accuracy={result['accuracy']:.3f}  "
        f"Precision={result['precision']:.3f}  "
        f"Recall={result['recall']:.3f}  "
        f"F1={result['f1']:.3f}"
    )


# ------------------------------------------------------------------
# 实验 1：为什么不能只看 Accuracy
# ------------------------------------------------------------------
imbalanced_y = torch.tensor([1] * 10 + [0] * 90)
always_normal = torch.zeros(100)
baseline = classification_metrics(imbalanced_y, always_normal)

print("【先看一个陷阱】100 条数据中只有 10 条正类，模型全部预测为负类")
show_metrics("全猜负类", baseline)
print(f"混淆矩阵：TP={baseline['tp']} FP={baseline['fp']} FN={baseline['fn']} TN={baseline['tn']}\n")


# ------------------------------------------------------------------
# 实验 2：生成数据，并且只划分一次
# ------------------------------------------------------------------
torch.manual_seed(42)

sample_count = 1000
x = torch.randn(sample_count, 2)
noise = 0.8 * torch.randn(sample_count)
y = (1.5 * x[:, 0] - x[:, 1] + noise > 0).float().reshape(-1, 1)

indices = torch.randperm(sample_count)
train_indices = indices[:600]
validation_indices = indices[600:800]
test_indices = indices[800:]

x_train, y_train = x[train_indices], y[train_indices]
x_validation, y_validation = x[validation_indices], y[validation_indices]
x_test, y_test = x[test_indices], y[test_indices]

print("【数据划分】")
print(f"train={len(x_train)}  validation={len(x_validation)}  test={len(x_test)}\n")


# ------------------------------------------------------------------
# 实验 3：训练 Logistic Regression
# ------------------------------------------------------------------
model = nn.Linear(2, 1)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epochs = 200

print("【训练过程】")
for epoch in range(1, epochs + 1):
    model.train()

    logits = model(x_train)
    loss = criterion(logits, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch in (1, 50, 100, 150, 200):
        train_result = classification_metrics(y_train, predict(model, x_train))
        validation_result = classification_metrics(
            y_validation,
            predict(model, x_validation),
        )
        print(
            f"epoch={epoch:>3}  loss={loss.item():.4f}  "
            f"train_F1={train_result['f1']:.3f}  "
            f"validation_F1={validation_result['f1']:.3f}"
        )


# ------------------------------------------------------------------
# 实验 4：用 validation 选择阈值，最后只评估一次 test
# ------------------------------------------------------------------
print("\n【只在 validation 上比较阈值】")
candidate_thresholds = (0.3, 0.5, 0.7)
validation_results = []

for threshold in candidate_thresholds:
    result = classification_metrics(
        y_validation,
        predict(model, x_validation, threshold),
    )
    validation_results.append((result["f1"], threshold, result))
    show_metrics(f"阈值 {threshold:.1f}", result)

_, best_threshold, best_validation_result = max(validation_results)

print(f"\nvalidation F1 最高的阈值：{best_threshold:.1f}")
test_result = classification_metrics(
    y_test,
    predict(model, x_test, best_threshold),
)

print("\n【最终结果】")
show_metrics("train", classification_metrics(y_train, predict(model, x_train, best_threshold)))
show_metrics("validation", best_validation_result)
show_metrics("test", test_result)
print(
    f"test 混淆矩阵：TP={test_result['tp']} FP={test_result['fp']} "
    f"FN={test_result['fn']} TN={test_result['tn']}"
)
