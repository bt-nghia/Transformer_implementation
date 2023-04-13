import torch
from torch import nn
import numpy as np


def positional_encoding(length, depth):
    depth = depth / 2
    positions = torch.arange(length)[:, np.newaxis]
    depths = torch.arange(depth)[np.newaxis, :]

    angle_rates = 1 / (10000 ** depths)
    angle_rads = positions * angle_rates

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1
    )

    return pos_encoding


class PositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.positional_encoding = positional_encoding(length=2048, depth=d_model)

    def compute_mask(self, *args, **kargs):
        return self.embedding(*args, **kargs)

    def forward(self, x):
        length = x.shape[1]
        x = self.embedding(x)

        x *= torch.sqrt(self.d_model)
        x += self.positional_encoding[None, :length, :]
        return x
