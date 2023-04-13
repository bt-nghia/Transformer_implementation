from model.Transformer import Transformer
from torchsummary import summary

num_layers = 6
d_model = 512
fc_hidden = 2048
num_heads = 8
drop_rate = 0.1
input_vocab_size = 32000  # 8500
output_vocab_size = 25000  # 7010

model = Transformer(vocab_size=input_vocab_size,
                    d_model=d_model,
                    n_head=num_heads,
                    fc_hidden=fc_hidden,
                    n_layer=num_layers,
                    trg_output=output_vocab_size)

summary(model)
