from torch import nn


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_head, fc_hidden, kdim=64, vdim=64):
        super(EncoderBlock, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=d_model,
                                                          num_heads=n_head,
                                                          kdim=kdim,
                                                          vdim=vdim)
        self.add_norm_1 = nn.LayerNorm(normalized_shape=d_model)

        self.feed_forward = nn.ModuleList([nn.Linear(in_features=d_model,
                                                     out_features=fc_hidden),
                                           nn.Linear(in_features=fc_hidden,
                                                     out_features=d_model)])
        self.add_norm_2 = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, x, s_mask):
        # sublayer 1
        x1 = x
        x = self.multi_head_attention(query=x, key=x, value=x)
        x = self.add_norm_1(x + x1)

        # sublayer 2
        x2 = x
        x = self.feed_forward(x)
        x = self.add_norm_2(x + x2)
        return x
