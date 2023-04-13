from torch import nn

from model.Decoder import Decoder
from model.Encoder import Encoder


class Transformer(nn.Module):
    def __init__(self, vocab_size, trg_output, d_model, fc_hidden, n_head, n_layer):
        super(Transformer, self).__init__()
        self.encoder = Encoder(vocab_size, d_model, n_head, fc_hidden, n_layer)
        self.decoder = Decoder(d_model, n_head, fc_hidden, n_layer)
        self.linear = nn.Linear(in_features=d_model, out_features=trg_output)

    def forward(self, x):
        context = self.encoder(x)
        x = self.decoder(x, context)
        x = self.linear(x)
        return x
