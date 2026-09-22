# -*- coding: utf-8 -*-
"""二分类指标和 validation threshold 选择。"""

import torch


def binary_metrics(probabilities, labels, threshold):
    probabilities = torch.as_tensor(probabilities)
    labels = torch.as_tensor(labels).long()
    predictions = (probabilities >= threshold).long()

    tp = int(((predictions == 1) & (labels == 1)).sum().item())
    tn = int(((predictions == 0) & (labels == 0)).sum().item())
    fp = int(((predictions == 1) & (labels == 0)).sum().item())
    fn = int(((predictions == 0) & (labels == 1)).sum().item())

    total = max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / total
    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def choose_threshold(probabilities, labels):
    candidates = [value / 100 for value in range(10, 91, 5)]
    results = [binary_metrics(probabilities, labels, value) for value in candidates]
    return max(results, key=lambda item: (item["f1"], item["recall"], -item["threshold"]))
