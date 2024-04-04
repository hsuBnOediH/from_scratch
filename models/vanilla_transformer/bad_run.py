import math

import torch
import torch.nn as nn
from torch.autograd import Variable


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


class ScaledDotProductAttention(nn.Module):
    def __init__(self, config, has_attention_mask=False):
        super().__init__()
        self.d_k = config.d_model // config.num_head
        self.seq_len = config.seq_len
        self.softmax = nn.Softmax(dim=-1)
        if has_attention_mask:
            single_head_attention_mask = torch.zeros(self.seq_len, self.seq_len)
            for i in range(self.seq_len):
                single_head_attention_mask[i, i + 1:] = float('-inf')
            single_head_attention_mask = single_head_attention_mask.unsqueeze(0)
            single_head_attention_mask = single_head_attention_mask.unsqueeze(0)
            self.attention_mask = single_head_attention_mask
        else:
            self.attention_mask = None
        self.num_head = config.num_head

    """
    multi_head_Q [bz * num_head * seq_len * d_q]
    multi_head_K [bz * num_head * seq_len * d_k]
    multi_head_V [bz * num_head * seq_len * d_v]
    """

    def forward(self, multi_head_q, multi_head_k, multi_head_v, padding_mask=None):
        # transpose the multi_head_K
        transpose_multi_head_k = torch.transpose(multi_head_k, -1, -2)
        # multi_head_Q * multi_head_K ^ T
        multi_head_score = torch.matmul(multi_head_q, transpose_multi_head_k) / self.d_k

        if self.attention_mask is not None:
            attention_mask = self.attention_mask.to(multi_head_score.device)
            multi_head_score += attention_mask

        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1)
            neg_inf = torch.tensor(float('-inf'))
            neg_inf = neg_inf.to(padding_mask.device)
            multi_head_score = torch.where(padding_mask == 1, multi_head_score, neg_inf)
        # scale
        d_k = torch.tensor(self.d_k)
        sqrt_d_k = torch.sqrt(d_k)
        scaled_multi_head_score = torch.div(multi_head_score, sqrt_d_k)
        # softmax score
        softmax_scaled_multi_head_score = self.softmax(scaled_multi_head_score)

        # retrieve Value
        # [bz * num_head * seq_len * d_v]
        multi_head_Z = torch.matmul(softmax_scaled_multi_head_score, multi_head_v)
        # transpose multi_head_Z
        transposed_multi_head_z = torch.transpose(multi_head_Z, -2, -3)
        # reshape [bz * seq_len * num_head * d_v]
        z = transposed_multi_head_z.reshape(transposed_multi_head_z.size(0), self.seq_len, -1)
        return z


class Attention(nn.Module):
    def __init__(self, config, is_decoder=False):
        super(Attention, self).__init__()
        d_model = config.d_model
        self.seq_len = config.seq_len
        self.num_head = config.num_head
        self.d_k = config.d_k
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        if is_decoder:
            self.scored_dot_product_att = ScaledDotProductAttention(config, has_attention_mask=True)
        else:
            self.scored_dot_product_att = ScaledDotProductAttention(config)

    def forward(self, q_input=None, k_input=None, v_input=None, padding_mask=None):
        if q_input is None:
            # error handling
            raise ValueError(" For Attention module, at least one of q_input should be provided")
        # x [batch_size, seq_len, d_model]
        # q, k, v [batch_size,seq_len,d_model]
        q = self.W_q(q_input)
        if k_input is None:
            k = self.W_k(q_input)
        else:
            k = self.W_k(k_input)

        if v_input is None:
            if k_input is None:
                v = self.W_v(q_input)
            else:
                v = self.W_v(k_input)
        else:
            v = self.W_v(v_input)
        cur_batch_size = q.size(0)
        multi_head_q = q.view(cur_batch_size, self.seq_len, self.num_head, self.d_k)
        multi_head_q = torch.transpose(multi_head_q, -2, -3)
        multi_head_k = k.view(cur_batch_size, self.seq_len, self.num_head, self.d_k)
        multi_head_k = torch.transpose(multi_head_k, -2, -3)

        multi_head_v = v.view(cur_batch_size, self.seq_len, self.num_head, self.d_k)
        multi_head_v = torch.transpose(multi_head_v, -2, -3)

        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1)
        output = self.scored_dot_product_att(multi_head_q, multi_head_k, multi_head_v, padding_mask)
        return output


class FeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super(FeedForwardNetwork, self).__init__()
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
        linear_1_res = self.liner1(x)
        active_res = nn.functional.relu(linear_1_res)
        dropout_res = self.dropout(active_res)
        linear_2_res = self.liner2(dropout_res)
        return linear_2_res


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        dropout_rate = config.dropout
        self.multi_head_attention = Attention(config)
        self.feed_forward_layer = FeedForwardNetwork(config)
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.dropout_2 = nn.Dropout(dropout_rate)

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

    def forward(self, input):
        x, padding_mask = input
        # copy an x for later add
        copy_x = torch.clone(x)
        # multi_head_attention sub layer
        multi_head_attention_res = self.multi_head_attention(x, padding_mask=padding_mask)
        # add and norm
        add_1_res = torch.add(multi_head_attention_res, copy_x)
        avg_1 = torch.mean(add_1_res, dim=1, keepdim=True)
        avg_zero_1_res = add_1_res - avg_1
        var_1 = torch.var(add_1_res, dim=1, keepdim=True)
        var_one_1_res = avg_zero_1_res / var_1
        # dropout
        sub_layer_1_res = self.dropout_1(var_one_1_res)

        # FFN sub layer
        copy_ffn_input = torch.clone(sub_layer_1_res)
        ffn_res = self.feed_forward_layer(sub_layer_1_res)
        # add and norm
        add_2_res = torch.add(ffn_res, copy_ffn_input)
        avg_2 = torch.mean(add_2_res, dim=1, keepdim=True)
        avg_zero_2_res = add_2_res - avg_2
        var_2 = torch.var(add_2_res)
        var_one_2_res = avg_zero_2_res / var_2
        # dropout

        sub_layer_2_res = self.dropout_2(var_one_2_res)

        return sub_layer_2_res, padding_mask


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder_layer_list = nn.Sequential()
        num_encoder_layer = config.encoder_layer
        for _ in range(num_encoder_layer):
            self.encoder_layer_list.append(EncoderLayer(config))

    def forward(self, x, padding_mask=None):
        res = self.encoder_layer_list((x, padding_mask))
        return res


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = Attention(config, is_decoder=True)
        self.cross_attention = Attention(config)
        self.ffn = FeedForwardNetwork(config)
        self.dropout_1 = nn.Dropout(config.dropout)
        self.dropout_2 = nn.Dropout(config.dropout)
        self.dropout_3 = nn.Dropout(config.dropout)

    def forward(self, input):
        encoder_last_output, prev_decoder_output, padding_mask = input
        copy_self_attention_input = torch.clone(prev_decoder_output)
        # masked multi head attention sub layer
        self_attention_res = self.self_attention(prev_decoder_output, padding_mask=padding_mask)
        # add and norm
        add_self_attention_res = self_attention_res + copy_self_attention_input
        avg_self_attention = add_self_attention_res.mean(dim=-2, keepdim=True)
        zero_avg_self_attention_res = add_self_attention_res - avg_self_attention

        var_self_attention = add_self_attention_res.var(dim=-2, keepdim=True)
        normalized_self_attention_res = zero_avg_self_attention_res / var_self_attention
        sub_layer_1_res = self.dropout_1(normalized_self_attention_res)

        # cross attention sub layer
        copy_cross_attention_input = torch.clone(sub_layer_1_res)
        cross_attention_res = self.cross_attention(sub_layer_1_res, encoder_last_output)
        # add and norm
        add_cross_attention_res = cross_attention_res + copy_cross_attention_input
        avg_cross_attention = add_cross_attention_res.mean(dim=-2, keepdim=True)
        zero_avg_cross_attention_res = add_cross_attention_res - avg_cross_attention

        var_cross_attention = add_cross_attention_res.var(dim=-2, keepdim=True)
        normalized_cross_attention_res = zero_avg_cross_attention_res / var_cross_attention
        sub_layer_2_res = self.dropout_2(normalized_cross_attention_res)

        # FFN sub layer
        copy_ffn_input = torch.clone(sub_layer_2_res)
        ffn_res = self.ffn(sub_layer_2_res)
        # add and norm
        add_ffn_res = ffn_res + copy_ffn_input
        avg_ffn = add_ffn_res.mean(dim=-2, keepdim=True)
        zero_avg_ffn_res = add_ffn_res - avg_ffn

        var_ffn = add_ffn_res.var(dim=-2, keepdim=True)
        normalized_ffn_res = zero_avg_ffn_res / var_ffn
        sub_layer_3_res = self.dropout_3(normalized_ffn_res)

        return encoder_last_output, sub_layer_3_res, padding_mask


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.decoder_layer_list = nn.Sequential()
        num_decoder_layer = config.encoder_layer
        for _ in range(num_decoder_layer):
            self.decoder_layer_list.append(
                DecoderLayer(config)
            )

    def forward(self, input):
        res = self.decoder_layer_list(input)
        return res


class EmbeddingLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        embedding_dim = config.d_model
        vocab_size = config.vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.d_model = config.d_model

    # the input of embedding layer is batch of token id
    # [batch_size * seq_len]
    def forward(self, input):
        res = self.embedding(input) * math.sqrt(self.d_model)
        return res


class PositionalEncodingLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        batch_size = config.batch_size
        seq_len = config.seq_len
        d_model = config.d_model

        pos = torch.arange(seq_len)
        pos = pos.unsqueeze(1)
        pos = pos.expand(seq_len, d_model)

        div_term = torch.arange(d_model).unsqueeze(0).expand(seq_len, d_model)
        div_term = div_term / d_model * torch.log(torch.tensor(10000.0))
        div_term = torch.exp(div_term)

        # pos / div_term
        inner = pos / div_term

        sin_term = torch.sin(inner)
        cos_term = torch.cos(inner)

        factor_0 = (torch.arange(d_model) + 1) % 2
        factor_1 = torch.arange(d_model) % 2

        # multiply sin_term with factor_0 and cos_term with factor_1
        pos_encoding = sin_term * factor_0 + cos_term * factor_1
        pos_encoding = pos_encoding.unsqueeze(0)
        pos_encoding = pos_encoding.expand(batch_size, -1, -1)
        pos_encoding = pos_encoding.to(config.device)
        self.register_buffer('pos_encoding', pos_encoding)

    # input is word embedding,
    # [batch_size * seq_len * d_model]
    def forward(self, embedding):
        # get the batch size of embedding
        # and match the pos_encoding batch size
        cur_batch_size = embedding.size(0)
        pos_encoding = self.pos_encoding[:cur_batch_size]
        return embedding + pos_encoding


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_layer = EmbeddingLayer(config)
        self.positional_encoding_layer = PositionalEncodingLayer(config)
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.linear = nn.Linear(config.d_model, config.vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    # input is batch of seq token
    # size [batch_size, seq_len]
    def forward(self, src_input, tgt_input, src_padding_mask=None, tgt_padding_mask=None):
        src_embedding = self.embedding_layer(src_input)
        tgt_embedding = self.embedding_layer(tgt_input)

        # embedding [batch_size, seq_len, d_model]
        src_pe_embedding = self.positional_encoding_layer(src_embedding)
        tgt_pe_embedding = self.positional_encoding_layer(tgt_embedding)

        # positional_encoded_embedding [batch_size * seq_len *
        encoder_output = self.encoder(src_pe_embedding, src_padding_mask)
        encoder_last_output, _ = encoder_output
        _, decoder_output, _ = self.decoder((encoder_last_output, tgt_pe_embedding, tgt_padding_mask))
        decoder_output = self.linear(decoder_output)
        logits = self.softmax(decoder_output)
        return logits