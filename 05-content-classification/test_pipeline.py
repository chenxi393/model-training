# -*- coding: utf-8 -*-
"""05 主链路的轻量测试。"""

import io
import tempfile
import unittest
from pathlib import Path

import torch
from fastapi.testclient import TestClient
from PIL import Image

from api import create_app
from generate_demo_data import SPLIT_SIZES, generate_dataset
from metrics import binary_metrics, choose_threshold
from model import create_model, load_model, save_checkpoint


class MetricsTest(unittest.TestCase):
    def test_binary_metrics(self):
        probabilities = torch.tensor([0.1, 0.8, 0.7, 0.2])
        labels = torch.tensor([0, 1, 0, 1])

        metrics = binary_metrics(probabilities, labels, threshold=0.5)

        self.assertEqual((metrics["tp"], metrics["tn"]), (1, 1))
        self.assertEqual((metrics["fp"], metrics["fn"]), (1, 1))
        self.assertAlmostEqual(metrics["f1"], 0.5)

    def test_choose_threshold_uses_labels(self):
        probabilities = torch.tensor([0.1, 0.4, 0.6, 0.9])
        labels = torch.tensor([0, 0, 1, 1])

        metrics = choose_threshold(probabilities, labels)

        self.assertEqual(metrics["f1"], 1.0)
        self.assertGreater(metrics["threshold"], 0.4)
        self.assertLessEqual(metrics["threshold"], 0.6)


class PipelineTest(unittest.TestCase):
    def test_demo_dataset_structure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generate_dataset(root)

            for split, count in SPLIT_SIZES.items():
                for label in ("normal", "unsafe"):
                    self.assertEqual(len(list((root / split / label).glob("*.png"))), count)

    def test_checkpoint_and_api_use_the_same_model(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            model = create_model(pretrained=False, freeze_backbone=True)
            save_checkpoint(
                model,
                checkpoint,
                class_to_idx={"normal": 0, "unsafe": 1},
                threshold=0.5,
                backbone="resnet18",
            )

            loaded_model, metadata = load_model(checkpoint)
            self.assertEqual(metadata["threshold"], 0.5)
            self.assertEqual(loaded_model.fc.out_features, 2)

            image = Image.new("RGB", (160, 160), (80, 140, 190))
            content = io.BytesIO()
            image.save(content, format="PNG")

            client = TestClient(create_app(checkpoint))
            response = client.post(
                "/predict",
                files={"file": ("sample.png", content.getvalue(), "image/png")},
            )

            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertIn(body["decision"], ("normal", "unsafe"))
            self.assertAlmostEqual(sum(body["probabilities"].values()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
