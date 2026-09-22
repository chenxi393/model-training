# -*- coding: utf-8 -*-
"""单张图片预处理和推理，共 CLI、Bad Case、FastAPI 使用。"""

import torch
from PIL import Image

from data import evaluation_transform


def predict_pil(model, image, metadata, device="cpu"):
    class_to_idx = metadata["class_to_idx"]
    index_to_class = {index: name for name, index in class_to_idx.items()}
    threshold = float(metadata["threshold"])

    image_tensor = evaluation_transform()(image.convert("RGB")).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu()

    probabilities_by_class = {
        index_to_class[index]: float(probability)
        for index, probability in enumerate(probabilities)
    }
    unsafe_probability = probabilities_by_class["unsafe"]
    decision = "unsafe" if unsafe_probability >= threshold else "normal"
    return {
        "decision": decision,
        "threshold": threshold,
        "probabilities": probabilities_by_class,
    }


def open_image(path):
    with Image.open(path) as image:
        return image.convert("RGB")
