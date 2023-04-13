import torch
from torch import nn
from block.DecoderBlock import DecoderBlock


class Decoder(nn.Module):
    def __init__(self, d_model, n_head, fc_hidden, n_layer):
        super(Decoder, self).__init__()

        self.decoder = nn.ModuleList([DecoderBlock(d_model=d_model,
                                                   fc_hidden=fc_hidden,
                                                   n_head=n_head)
                                      for _ in range(n_layer)])

    def forward(self, x, context):
        # handle ouput from encoder
        # 2nd sublayer input: K, V from encoder, Q from 1st sublayer

        for layer in self.decoder:
            x = layer(x)

        return x
