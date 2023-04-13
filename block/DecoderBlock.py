from torch import nn


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_head, fc_hidden, kdim=64, vdim=64):
        super(DecoderBlock, self).__init__()
        # masked attention
        self.masked_multi_head_attention = nn.MultiheadAttention(embed_dim=d_model,
                                                                 num_heads=n_head,
                                                                 kdim=kdim,
                                                                 vdim=vdim)
        self.add_norm1 = nn.LayerNorm(normalized_shape=d_model)

        # cross attention
        self.multihead_attention = nn.MultiheadAttention(embed_dim=d_model,
                                                         num_heads=n_head,
                                                         kdim=kdim,
                                                         vdim=vdim)
        self.add_norm2 = nn.LayerNorm(normalized_shape=d_model)

        self.feed_forward = nn.ModuleList([nn.Linear(in_features=d_model,
                                                     out_features=fc_hidden),
                                           nn.Linear(in_features=fc_hidden,
                                                     out_features=d_model)])
        self.add_norm3 = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, x, context):
        # sublayer 1
        x1 = x
        x = self.masked_multi_head_attention(query=x, key=x, value=x, key_padding_mask=True)
        x = self.add_norm1(x + x1)

        # sublayer 2: cross multihead attention
        x2 = x
        x = self.multihead_attention(query=x, key=context, value=context)
        x = self.add_norm2(x + x2)

        # sublayer 3
        x3 = x
        x = self.feed_forward(query=x, key=x, value=x)
        x = self.add_norm3(x + x3)
        return x
