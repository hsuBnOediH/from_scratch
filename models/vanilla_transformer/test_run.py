from transformer_structure import *
import torch

BATCH_SIZE = 2
ENCODER_LAYER = 6
DECODER_LAYER = 6
MAX_SEQ_LEN = 128
MODEL_SIZE = 256
HIDDEN_SIZE = 512
DROP_OUT = 0.1
NUM_HEAD = 8

config = TransformerConfig(
    batch_size=BATCH_SIZE,
    seq_len=MAX_SEQ_LEN,
    encoder_layer=ENCODER_LAYER,
    decoder_layer=DECODER_LAYER,
    d_model=MODEL_SIZE,
    hidden_size=HIDDEN_SIZE,
    dropout=DROP_OUT,
    num_head=NUM_HEAD)


def test_scaled_dot_product_attention():
    print("Testing Scaled Dot Product Attention")
    attention = ScaledDotProductAttention(config)
    multi_head_q = torch.randn(config.batch_size, config.num_head,
                               config.seq_len, config.d_k)
    multi_head_k = torch.randn(config.batch_size, config.num_head,
                               config.seq_len, config.d_k)
    multi_head_v = torch.randn(config.batch_size, config.num_head,
                               config.seq_len, config.d_k)

    print(f" Input multi_head_q shape: {multi_head_q.shape}")
    print(f" Input multi_head_k shape: {multi_head_k.shape}")
    print(f" Input multi_head_v shape: {multi_head_v.shape}")

    res = attention(multi_head_q, multi_head_k, multi_head_v)
    print(f" Output shape: {res.shape}")
    assert res.shape == (
        config.batch_size, config.seq_len, config.d_model), f"Output shape is not as expected: {res.shape}"


def test_feed_forward_network():
    print("Testing Feed Forward Network")
    ff = FeedForwardNetwork(config)
    x = torch.randn(config.batch_size, config.seq_len, config.d_model)
    print(f" Input shape: {x.shape}")
    res = ff(x)
    print(f" Output shape: {res.shape}")
    assert res.shape == (
        config.batch_size, config.seq_len, config.d_model), f"Output shape is not as expected: {res.shape}"


def test_self_attention():
    print("Testing Self Attention")
    attention = SelfAttention(config)
    x = torch.randn(config.batch_size, config.seq_len, config.d_model)
    print(f" Input shape: {x.shape}")
    res = attention(x)
    print(f" Output shape: {res.shape}")
    assert res.shape == (
        config.batch_size, config.seq_len, config.d_model), f"Output shape is not as expected: {res.shape}"


def test_encoder_layer():
    print("Testing Encoder Layer")
    encoder_layer = EncoderLayer(config)
    x = torch.randn(config.batch_size, config.seq_len, config.d_model)
    print(f" Input shape: {x.shape}")
    res = encoder_layer(x)
    print(f" Output shape: {res.shape}")
    assert res.shape == (
        config.batch_size, config.seq_len, config.d_model), f"Output shape is not as expected: {res.shape}"


def test_encoder():
    print("Testing Encoder")
    encoder = Encoder(config)
    x = torch.randn(config.batch_size, config.seq_len, config.d_model)
    print(f" Input shape: {x.shape}")
    res = encoder(x)
    print(f" Output shape: {res.shape}")
    assert res.shape == (
        config.batch_size, config.seq_len, config.d_model), f"Output shape is not as expected: {res.shape}"
