import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# Test the debugged implementation of the transformer model
# res works well, 3.06 --> 2.2

# BATCH_SIZE = 32 if device_type == "mps" else 64
# SEQ_LEN = 64 if device_type == "mps" else 512
# ENCODER_LAYER = 6
# DECODER_LAYER = 6
# D_MODEL = 256 if device_type == "mps" else 512
# HIDDEN_DIM = 512 if device_type == "mps" else 2048
# NUM_HEADS = 8
# DROPOUT = 0.1
# VOCAB_SIZE = len(token_to_id)
# EPOCHS = 30
# STEPS = 1000000
# BETA1 = 0.9
# BETA2 = 0.98
# EPSILON = 1e-9
# LEARNING_RATE = 0.00001
# WARMUP_STEPS = 4000


# 1. use own implementation of the multi-head attention




def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TransformerConfig:
    def __init__(self, batch_size, seq_len=256,
                 encoder_layer=6, decoder_layer=6, d_model=512, num_head=8,
                 hidden_size=2048, dropout=0.1, vocab_size=50000, device="cpu",
                 eps=1e-6):
        self.batch_size = batch_size
        self.encoder_layer = encoder_layer
        self.decoder_layer = decoder_layer
        self.d_model = d_model
        self.num_head = num_head
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.d_k = d_model // num_head
        self.vocab_size = vocab_size
        self.device = device
        self.eps = eps

        assert d_model % num_head == 0, "d_model should be divided by num_head "


"""
Without batch size:
    Input: X, size is  [seq_len * d_model]
    From the perspective of each head, the computation is following
    
    X [seq_len * d_model] *  the first Head Weight W_q_0 [d_model * d_q]
    Then We get the Q_0 [seq_len * d_q], where 0 is the index for the first head
    
    X [seq_len * d_model] * Head Weight W_k_0 [d_model * d_k], resulted K_0 [seq_len * d_k]

    X [seq_len * d_model] * Head Weight W_v_0 [d_model * d_v], resulted V_0 [seq_len * d_v]
    
    Since we want Q_0 and K_0^T to mul to give us a Score tensor [seq_len * seq_len], d_q has to equal to d_k
    
    Then we could use the Score tensor [seq_len * seq_len] to retrieve the V_0 according to those weight in Score
    formerly,  Score tensor [seq_len * seq_len] * V_0 [seq_len * d_v], result in Z_0 [seq_len * d_v] 
    (since softmax and divide by sqrt(d_k) won't change the dim, let's ignore them for now)
    
    From a multi_head perspective, since we are going to apply the since attention operation  many times (6,12 depend on
     the number of encode/decoder layer)
    We want to maintain the same dim form input X and final output Z
    which indicate we have to find a way to assembly those Z_i [seq_len * d_v] togather,where i is the index of header,
    make Z has the same dim of X [seq_len * d_model], the simplest way come to one's head is stack those Z_i togather. 
    d_v = d_model / num_head
    
    
    Put all the head togather, we got X [seq_len * d_model] * Head Weight W_q [num_head * d_model * d_q], resulted Q 
    [num_head * seq_len * d_q] But this can't be done directly, since the dim of X is [seq_len * d_model], 
    we have to expand the dim of X to [num_head * seq_len * d_model] So the final calculation is X_expand [ num_head 
    * seq_len * d_model] * Head Weight W_q [num_head * d_model * d_q], resulted multi_head_K [num_head * seq_len * d_q]
    
    same for the K and V X_expand [ num_head * seq_len * d_model] will mul the W_k [num_head * d_model * d_k] to get 
    multi_head_K [num_head * seq_len * d_k] X_expand [ num_head * seq_len * d_model] will mul the W_v [num_head * 
    d_model * d_v] to get multi_head_V [num_head * seq_len * d_v]
    
    multi_head Score [num_head * seq_len * seq_len] could be computed by mul multi_head_Q [num_head * seq_len * d_q] 
    and multi_head_K^T [num_head * seq_len * d_k]
    
    then we can use the multi_head Score to retrieve the multi_head_V multi_head Score [num_head * seq_len * seq_len] 
    *  multi_head_V [num_head * seq_len * d_v], we got multi_head_Z [ num_head, seq_len * d_v]
    
    Now we have the foundation, since multi_head_Z and Z has the same number of digits. We could transpose the dim 
    back and stack the multi_head_Z togather to get the final output Z [seq_len * d_model] multi_head_Z [ num_head, 
    seq_len * d_v] -> transpose_multi_head_Z [seq_len * num_head * d_v] -> Z [seq_len * d_model]
    
With batch size: Now we can add the batch size at beginning of the all the tensor, for all the tensor, the first dim 
    is batch size and all the step won't be affected by the batch size change.
    
    the whole process could be described as: 1. expand input X: X [bz * seq_len * d_model] -> X_expand [bz * num_head 
    * seq_len * d_model] 2. compute multi_head_Q, multi_head_K, multi_head_V X_expand [bz * num_head * seq_len * 
    d_model] * W_q [num_head * d_model * d_q] -> multi_head_Q [bz * num_head * seq_len * d_q] X_expand [bz * num_head 
    * seq_len * d_model] * W_k [num_head * d_model * d_k] -> multi_head_K [bz * num_head * seq_len * d_k] X_expand [
    bz * num_head * seq_len * d_model] * W_v [num_head * d_model * d_v] -> multi_head_V [bz * num_head * seq_len * 
    d_v] 3. compute multi_head Score multi_head_Q [bz * num_head * seq_len * d_q] * multi_head_K^T [bz * num_head * 
    seq_len * d_k] -> multi_head Score [bz * num_head * seq_len * seq_len] 4. compute multi_head_Z multi_head Score [
    bz * num_head * seq_len * seq_len] * multi_head_V [bz * num_head * seq_len * d_v] -> multi_head_Z [bz * num_head 
    * seq_len * d_v] 5. transpose and stack multi_head_Z multi_head_Z [bz * num_head * seq_len * d_v] -> 
    transpose_multi_head_Z [bz * seq_len * num_head * d_v] -> Z [bz * seq_len * d_model]"""


def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    d_k = q.size(-1)
    # [batch_size, head_num, seq_len, seq_len]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill_.html#torch.Tensor.masked_fill_
        scores = scores.masked_fill_(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # return the weighted value and the attention weights
    return torch.matmul(p_attn, v), p_attn
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()

        self.d_k = config.d_k
        self.num_head = config.num_head
        self.d_model = config.d_model
        self.linears = clones(nn.Linear(config.d_model, config.d_model), 4)
        self.dropout = nn.Dropout(p=config.dropout)
        self.seq_len = config.seq_len
        # todo
        self.attn = None

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # TODO
            mask = mask.unsqueeze(1)
            mask = mask.unsqueeze(-1)
        num_batch = query.size(0)

        query, key, value = [l(x).view(num_batch, -1, self.num_head, self.d_k).transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))]
        x, self.attn = scaled_dot_product_attention(query, key, value, mask=mask, dropout=self.dropout)
        # if the transpose is in place, the raw data(1-dimensional) won't be changed
        # only the meta data will be changed, the data_ptr is the same
        # only use contiguous() to make sure the raw data is changed
        x = x.transpose(1, 2).contiguous().view(num_batch, -1, self.d_model)
        # according to the paper, the output of multi head attention will be passed through a linear layer
        return self.linears[-1](x)




class PositionwiseFeedForward(nn.Module):
    def __init__(self, config):
        super(PositionwiseFeedForward, self).__init__()
        hidden_size = config.hidden_size
        d_model = config.d_model
        self.liner1 = nn.Linear(d_model, hidden_size)
        self.liner2 = nn.Linear(hidden_size, d_model)
        self.dropout = nn.Dropout(config.dropout)

    """
    the input of FFN will be the output of attention module, 
    since the attention module won't change the dim of input,
    it will still be [batch_size * seq_len * d_model]
    
    In this FeedForwardNetwork, the input will pass the linear layer 1 and extend the dim
    and go through the sec liner layer change the dim back, keep the dim same for future layer
    """

    def forward(self, x):
        return self.liner2(self.dropout(F.relu(self.liner1(x))))


class SublayerConnection(nn.Module):
    def __init__(self, config):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, sublayer):
        # the input will be normed first and then pass to sublayer
        # then the dropout will be applied
        # finally, the output will be added to the input
        return x + self.dropout(sublayer(self.norm(x)))


class Embeddings(nn.Module):
    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.d_model = config.d_model
        vocab = config.vocab_size
        self.lut = nn.Embedding(vocab, self.d_model)

    # the input of embedding layer is batch of token id
    # [batch_size * seq_len]
    def forward(self, input):
        res = self.lut(input) * math.sqrt(self.d_model)
        return res


class LayerNorm(nn.Module):
    def __init__(self, config):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(config.d_model))
        self.b_2 = nn.Parameter(torch.zeros(config.d_model))
        self.eps = config.eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionalEncoding(nn.Module):

    def __init__(self, config):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=config.dropout)
        pe = torch.zeros(config.seq_len, config.d_model)
        position = torch.arange(0, config.seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.d_model, 2) * -(math.log(10000.0) / config.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    # input is word embedding,
    # [batch_size * seq_len * d_model]
    def forward(self, embedding):
        # get the batch size of embedding
        # and match the pos_encoding batch size
        x = embedding + Variable(self.pe[:, :embedding.size(1)], requires_grad=False)
        return self.dropout(x)


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(config)
        self.feed_forward = PositionwiseFeedForward(config)
        self.sublayer = clones(SublayerConnection(config), 2)

    """
    For Encoder Layer, what the it's the first layer or not, all the input will be the same dim
    [bath_size * seq_len * d_model]
    
    For the first layer, the input is output of embedding layer
    
    For others, since the encoder layer always keep the input and output dim the same. So the EncodeLayer is reusable.
    
    
    The EncodeLayer will have two sub layers, 
    1. multi-head attention 
    2. feed forward network
    After each sub layer, there will be two actions be processed.
    1. add and norm
    2. drop out
    """

    def forward(self, x, padding_mask):
        # the lambda create a temporary function to pass the x to the multi_head_attention with the padding_mask
        x = self.sublayer[0](x, lambda x: self.multi_head_attention(x, x, x, padding_mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_list = nn.ModuleList(
            [copy.deepcopy(EncoderLayer(config)) for _ in range(config.encoder_layer)])
        self.layer_norm = LayerNorm(config)

    def forward(self, x, padding_mask=None):
        for layer in self.layer_list:
            x = layer(x, padding_mask)
        return self.layer_norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(config)
        self.cross_attention = MultiHeadAttention(config)
        self.ffn = PositionwiseFeedForward(config)
        self.sublayer = clones(SublayerConnection(config), 3)

    def forward(self, x, memory, src_padding_mask, tgt_padding_mask):
        x = self.sublayer[0](x, lambda x: self.self_attention(x, x, x, tgt_padding_mask))
        x = self.sublayer[1](x, lambda x: self.cross_attention(x, memory, memory, src_padding_mask))
        return self.sublayer[2](x, self.ffn)


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.decoder_layer_list = nn.ModuleList(
            [copy.deepcopy(DecoderLayer(config)) for _ in range(config.decoder_layer)]
        )
        self.norm = LayerNorm(config)

    def forward(self, x, encoder_last_output, src_padding_mask, tgt_padding_mask):
        for layer in self.decoder_layer_list:
            x = layer(encoder_last_output, x, src_padding_mask, tgt_padding_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.embedding = Embeddings(config)
        self.positional_encoding = PositionalEncoding(config)
        self.proj = nn.Linear(config.d_model, config.vocab_size)

    # input is batch of seq token
    # size [batch_size, seq_len]
    def forward(self, src_input, tgt_input, src_padding_mask=None, tgt_padding_mask=None):
        src_embedding = self.embedding(src_input)
        tgt_embedding = self.embedding(tgt_input)
        src_pe_embedding = self.positional_encoding(src_embedding)
        tgt_pe_embedding = self.positional_encoding(tgt_embedding)

        memory = self.encoder(src_pe_embedding, src_padding_mask)
        decoder_output = self.decoder(tgt_pe_embedding, memory, src_padding_mask, tgt_padding_mask)
        logits = F.log_softmax(self.proj(decoder_output), dim=-1)
        return logits
