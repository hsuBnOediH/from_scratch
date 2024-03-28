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
VOCAB_SIZE = 50000


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


def test_dot_product_attention_with_mask():
    print(
        "Testing Dot Product Attention with Mask"
    )
    attention = ScaledDotProductAttention(config, attention_mask=True)
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
    attention = Attention(config)
    x = torch.randn(config.batch_size, config.seq_len, config.d_model)
    print(f" Input shape: {x.shape}")
    res = attention(x)
    print(f" Output shape: {res.shape}")
    assert res.shape == (
        config.batch_size, config.seq_len, config.d_model), f"Output shape is not as expected: {res.shape}"

def test_cross_attention():
    print("Testing Cross Attention")
    attention = Attention(config)
    x = torch.randn(config.batch_size, config.seq_len, config.d_model)
    y = torch.randn(config.batch_size, config.seq_len, config.d_model)
    print(f" Input shape: {x.shape}")
    res = attention(x, y)
    print(f" Output shape: {res.shape}")
    assert res.shape == (
        config.batch_size, config.seq_len, config.d_model), f"Output shape is not as expected: {res.shape}"

def test_self_attention_with_mask():
    print("Testing Self Attention with Mask")
    attention = Attention(config, is_decoder=True)
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

def test_decoder_layer():
    print("Testing Decoder Layer")
    decoder_layer = DecoderLayer(config)
    x = torch.randn(config.batch_size, config.seq_len, config.d_model)
    memory = torch.randn(config.batch_size, config.seq_len, config.d_model)
    print(f" Input shape: {x.shape}")
    decoder_layer, memery_output = decoder_layer((x, memory))

    # assert the output shape of decoder layer
    assert decoder_layer.shape == (
        config.batch_size, config.seq_len, config.d_model), f"Output shape is not as expected: {decoder_layer.shape}"
    # assert the memory output shape of decoder layer is the same and the content is the same as the input memory
    assert torch.equal(memory, memery_output), "Memory output is not the same as the input memory"

def test_decoder():
    print("Testing Decoder")
    decoder = Decoder(config)
    x = torch.randn(config.batch_size, config.seq_len, config.d_model)
    memory = torch.randn(config.batch_size, config.seq_len, config.d_model)
    print(f" Input shape: {x.shape}")
    res,memory_output= decoder(x, memory)
    print(f" Output shape: {res.shape}")

    # assert the output shape of decoder layer
    assert res.shape == (
        config.batch_size, config.seq_len, config.d_model), f"Output shape is not as expected: {res.shape}"

    # assert the memory output shape of decoder layer is the same and the content is the same as the input memory
    assert torch.equal(memory, memory_output), "Memory output is not the same as the input memory"


def test_embedding_layer():
    print("Testing Embedding Layer")
    embedding_layer = EmbeddingLayer(config)
    x = torch.randint(0, VOCAB_SIZE, (config.batch_size, config.seq_len))
    print(f" Input shape: {x.shape}")
    res = embedding_layer(x)
    print(f" Output shape: {res.shape}")
    assert res.shape == (
        config.batch_size, config.seq_len, config.d_model), f"Output shape is not as expected: {res.shape}"

def test_positional_encoding_layer():
    print("Testing Positional Encoding Layer")
    positional_encoding_layer = PositionalEncodingLayer(config)
    x = torch.randn(config.batch_size, config.seq_len, config.d_model)
    print(f" Input shape: {x.shape}")
    res = positional_encoding_layer(x)
    print(f" Output shape: {res.shape}")
    assert res.shape == (
        config.batch_size, config.seq_len, config.d_model), f"Output shape is not as expected: {res.shape}"


def test_transformer():
    print("Testing Transformer")
    transformer = Transformer(config)
    x = torch.randint(0, VOCAB_SIZE, (config.batch_size, config.seq_len))
    y = torch.randint(0, VOCAB_SIZE, (config.batch_size, config.seq_len))
    print(f" Input shape: {x.shape}")
    res = transformer(x, y)
    print(f" Output shape: {res.shape}")
    assert res.shape == (
        config.batch_size, config.seq_len, config.d_model), f"Output shape is not as expected: {res.shape}"



def test_transformer_forward_and_backward():
    print("Testing Transformer forward and backward")
    transformer = Transformer(config)
    x = torch.randint(0, VOCAB_SIZE, (config.batch_size, config.seq_len))
    y = torch.randint(0, VOCAB_SIZE, (config.batch_size, config.seq_len))
    print(f" Input shape: {x.shape}")
    res = transformer(x, y)
    # compute loss and backprop
    loss = res.sum()
    loss.backward()
    print(f" Output shape: {res.shape}")
    print(f" Loss: {loss}")
    assert res.shape == (
        config.batch_size, config.seq_len, config.d_model), f"Output shape is not as expected: {res.shape}"
