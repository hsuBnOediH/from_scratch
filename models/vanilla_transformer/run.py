import torch

from transformer_structure import *


BATCH_SIZE = 2
ENCODER_LAYER = 6
DECODER_LAYER = 6
MAX_SEQ_LEN = 128
HIDDEN_SIZE = 256
NUM_HEAD = 8


config = TransformerConfig(
    batch_size=BATCH_SIZE,
    seq_len=MAX_SEQ_LEN,
    encoder_layer=ENCODER_LAYER,
    decoder_layer=DECODER_LAYER,
    d_model=HIDDEN_SIZE,
    num_head=NUM_HEAD)


# attention = ScaledDotProductAttention(config)
# multi_head_q = torch.randn(config.batch_size, config.num_head,
#                            config.seq_len,config.d_k)
# multi_head_k = torch.randn(config.batch_size, config.num_head,
#                            config.seq_len,config.d_k)
# multi_head_v = torch.randn(config.batch_size, config.num_head,
#                            config.seq_len,config.d_k)
#
# print(multi_head_q.shape)
# print(multi_head_k.shape)
# print(multi_head_v.shape)
# res  = attention(multi_head_q,multi_head_k,multi_head_v)
#
# print(res.shape)

attention = SelfAttention(config)
x = torch.randn(config.batch_size, config.seq_len, config.d_model)
print(x.shape)
res = attention(x)
print(res.shape)