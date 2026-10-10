# -*- coding: utf-8 -*-
"""tiny Transformer 的核心行为测试。"""

import unittest

import torch

from data import CharTokenizer, sample_batch
from model import TinyTransformer, TransformerConfig


class TokenizerTest(unittest.TestCase):
    def test_round_trip(self):
        tokenizer = CharTokenizer.from_text("模型学习")
        text = "模型学习"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)


class DataTest(unittest.TestCase):
    def test_targets_are_shifted_one_token(self):
        data = torch.arange(30)
        generator = torch.Generator().manual_seed(42)
        inputs, targets = sample_batch(
            data,
            batch_size=4,
            context_length=8,
            device="cpu",
            generator=generator,
        )
        self.assertTrue(torch.equal(inputs[:, 1:], targets[:, :-1]))


class TransformerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = TransformerConfig(
            vocab_size=12,
            context_length=8,
            d_model=16,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
        )
        self.model = TinyTransformer(self.config).eval()

    def test_forward_shapes_and_loss(self):
        inputs = torch.randint(0, self.config.vocab_size, (3, 8))
        targets = torch.randint(0, self.config.vocab_size, (3, 8))
        logits, loss, attentions = self.model(
            inputs,
            targets,
            return_attention=True,
        )

        self.assertEqual(tuple(logits.shape), (3, 8, self.config.vocab_size))
        self.assertEqual(len(attentions), self.config.num_layers)
        self.assertEqual(tuple(attentions[0].shape), (3, 4, 8, 8))
        self.assertGreater(loss.item(), 0)

    def test_future_tokens_do_not_change_past_logits(self):
        first = torch.tensor([[1, 2, 3, 4, 5]])
        second = torch.tensor([[1, 2, 3, 9, 10]])

        first_logits, _, _ = self.model(first)
        second_logits, _, _ = self.model(second)

        self.assertTrue(
            torch.allclose(
                first_logits[:, :3],
                second_logits[:, :3],
                atol=1e-6,
            )
        )

    def test_generate_adds_requested_tokens(self):
        prompt = torch.tensor([[1, 2, 3]])
        generated = self.model.generate(prompt, max_new_tokens=5)
        self.assertEqual(tuple(generated.shape), (1, 8))


if __name__ == "__main__":
    unittest.main()
