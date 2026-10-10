# -*- coding: utf-8 -*-
"""一个用于学习的 tiny causal Transformer。"""

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_CHECKPOINT = Path(__file__).parent / "artifacts" / "tiny_transformer.pt"


@dataclass
class TransformerConfig:
    vocab_size: int
    context_length: int = 48
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1

    def __post_init__(self):
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除")


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_size = config.d_model // config.num_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(
                    config.context_length,
                    config.context_length,
                    dtype=torch.bool,
                )
            ),
        )

    def forward(self, x):
        batch_size, sequence_length, d_model = x.shape
        query, key, value = self.qkv(x).chunk(3, dim=-1)

        def split_heads(tensor):
            return tensor.view(
                batch_size,
                sequence_length,
                self.num_heads,
                self.head_size,
            ).transpose(1, 2)

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)

        scores = query @ key.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_size)
        mask = self.causal_mask[:sequence_length, :sequence_length]
        scores = scores.masked_fill(~mask, float("-inf"))
        attention_weights = F.softmax(scores, dim=-1)

        attended_values = self.attention_dropout(attention_weights) @ value
        attended_values = attended_values.transpose(1, 2).contiguous()
        attended_values = attended_values.view(batch_size, sequence_length, d_model)
        return self.output_dropout(self.output(attended_values)), attention_weights


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = nn.LayerNorm(config.d_model)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        attention_output, attention_weights = self.attention(
            self.attention_norm(x)
        )
        x = x + attention_output
        x = x + self.feed_forward(self.feed_forward_norm(x))
        return x, attention_weights


class TinyTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(
            config.context_length,
            config.d_model,
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.language_model_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

    def forward(self, input_ids, targets=None, return_attention=False):
        _, sequence_length = input_ids.shape
        if sequence_length > self.config.context_length:
            raise ValueError(
                f"序列长度 {sequence_length} 超过 context_length "
                f"{self.config.context_length}"
            )

        positions = torch.arange(sequence_length, device=input_ids.device)
        x = self.token_embedding(input_ids)
        x = x + self.position_embedding(positions)

        attentions = []
        for block in self.blocks:
            x, attention_weights = block(x)
            if return_attention:
                attentions.append(attention_weights)

        logits = self.language_model_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
            )
        return logits, loss, attentions

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0):
        if temperature <= 0:
            raise ValueError("temperature 必须大于 0")

        self.eval()
        for _ in range(max_new_tokens):
            visible_ids = input_ids[:, -self.config.context_length :]
            logits, _, _ = self(visible_ids)
            next_token_logits = logits[:, -1, :] / temperature
            probabilities = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
        return input_ids


def save_checkpoint(model, tokenizer, path=DEFAULT_CHECKPOINT, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(model.config),
            "characters": tokenizer.characters,
            "metadata": metadata,
        },
        path,
    )
    return path


def load_checkpoint(path=DEFAULT_CHECKPOINT, device="cpu"):
    from data import CharTokenizer

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = TransformerConfig(**checkpoint["config"])
    model = TinyTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    tokenizer = CharTokenizer(checkpoint["characters"])
    return model, tokenizer, checkpoint.get("metadata", {})
