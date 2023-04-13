import torch

from torch import nn
from block.EncoderBlock import EncoderBlock
from block.PositionalEmbedding import PositionalEmbedding


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, fc_hidden, n_layer):
        super(Encoder, self).__init__()

        self.embed = PositionalEmbedding(vocab_size=vocab_size,
                                         d_model=d_model)

        self.encoder = nn.ModuleList([EncoderBlock(d_model=d_model,
                                                   fc_hidden=fc_hidden,
                                                   n_head=n_head)
                                      for _ in range(n_layer)])

    def forward(self, x):
        x = self.embed(x)

        for layer in self.encoder:
            x = layer(x)

        return x
