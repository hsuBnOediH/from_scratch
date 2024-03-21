import torch
import torch.nn as nn


class TransformerConfig:
    def __init__(self, batch_size, seq_len=256,
                 encoder_layer=6, decoder_layer=6, d_model=512, num_head=8):
        self.batch_size = batch_size
        self.encoder_layer = encoder_layer
        self.decoder_layer = decoder_layer
        self.d_model = d_model
        self.num_heads = num_head
        self.seq_len = seq_len
        self.d_k = d_model // num_head

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
    def __init__(self, config):
        super().__init__()
        self.d_k = config.d_model // config.num_head
        self.batch_size = config.batch_size
        self.max_len = config.max_len
        self.softmax = nn.Softmax(dim=-1)

    """
    multi_head_Q [bz * num_head * seq_len * d_q]
    multi_head_K [bz * num_head * seq_len * d_k]
    multi_head_V [bz * num_head * seq_len * d_v]
    """

    def forward(self, multi_head_q, multi_head_k, multi_head_v):
        # transpose the multi_head_K
        transpose_multi_head_k = torch.transpose(multi_head_k, -1, -2)
        # multi_head_Q * multi_head_K ^ T
        multi_head_score = torch.matmul(multi_head_q, transpose_multi_head_k)
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
        z = transposed_multi_head_z.view(self.batch_size, self.max_len, -1)
        return z


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config.d_model
        self.batch_size = config.batch_size
        self.seq_len = config.seq_len
        self.num_head = config.num_head
        self.d_k = config.d_k
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.scored_dot_product_att = ScaledDotProductAttention(config)

    def forward(self, x):
        # x [batch_size, seq_len, d_model]
        # q, k, v [batch_size,seq_len,d_model]
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        multi_head_q = q.view(self.batch_size, self.seq_len, self.num_head, self.d_k)
        multi_head_q = torch.transpose(multi_head_q, -2, -3)
        multi_head_k = k.view(self.batch_size, self.seq_len, self.num_head, self.d_k)
        multi_head_k = torch.transpose(multi_head_k, -2, -3)

        multi_head_v = v.view(self.batch_size, self.seq_len, self.num_head, self.d_k)
        multi_head_v = torch.transpose(multi_head_v, -2, -3)

        output = self.scored_dot_product_att(multi_head_q, multi_head_k, multi_head_v)
        return output
