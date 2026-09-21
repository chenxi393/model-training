# -*- coding: utf-8 -*-
"""第四章模型、checkpoint 和图片预处理的最小测试。"""

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image, ImageDraw


CHAPTER_DIR = Path(__file__).resolve().parent.parent
if str(CHAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(CHAPTER_DIR))

from model import MLP, load_model, save_checkpoint
from predict_image import normalize_image, predict, prepare_image


class MLPTest(unittest.TestCase):
    def test_model_outputs_ten_probabilities(self):
        model = MLP()
        images = torch.zeros(2, 28 * 28)

        predicted_digit, probabilities = predict(model, images[:1])

        self.assertIn(predicted_digit, range(10))
        self.assertEqual(probabilities.shape, (10,))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=5)

    def test_checkpoint_round_trip_keeps_predictions(self):
        torch.manual_seed(42)
        model = MLP().eval()
        image = torch.rand(1, 28 * 28)
        expected = model(image)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "model.pt"
            save_checkpoint(model, checkpoint_path, test_accuracy=0.9)
            loaded_model, metadata = load_model(checkpoint_path)

        self.assertTrue(torch.equal(expected, loaded_model(image)))
        self.assertEqual(metadata["test_accuracy"], 0.9)

    def test_prepare_image_returns_mnist_shape_and_range(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "digit.png"
            image = Image.new("L", (80, 100), color=255)
            draw = ImageDraw.Draw(image)
            draw.line((40, 15, 40, 85), fill=0, width=10)
            image.save(image_path)

            tensor = prepare_image(image_path)

        self.assertEqual(tensor.shape, (1, 28 * 28))
        self.assertGreater(float(tensor.max()), 0.0)
        self.assertGreaterEqual(float(tensor.min()), 0.0)
        self.assertLessEqual(float(tensor.max()), 1.0)

    def test_standard_mnist_image_is_not_resized_again(self):
        image = Image.new("L", (28, 28), color=0)
        image.putpixel((4, 9), 255)

        normalized = normalize_image(image)

        self.assertEqual(normalized.size, (28, 28))
        self.assertEqual(normalized.getpixel((4, 9)), 255)
        self.assertEqual(normalized.getbbox(), (4, 9, 5, 10))


if __name__ == "__main__":
    unittest.main()
