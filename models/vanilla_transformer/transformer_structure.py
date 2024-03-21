import torch.nn as nn
class TransformerConfig():
    def __init__(self, batch_size, seq_len = 256,
                 encoder_layer = 6, decoder_layer = 6, d_model = 512, num_heads = 8):
        self.batch_size = batch_size
        self.encoder_layer = encoder_layer
        self.decoder_layer = decoder_layer
        self.d_model = d_model
        self.num_heads = num_heads
        self.seq_len = seq_len

        assert d_model % num_heads == 0, "d_model should be divided by num_head "



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
    
    From a multi-head perspective, since we are going to apply the since attention operation  many times (6,12 depend on the number of encode/decoder layer)
    We want to maintain the same dim form input X and final output Z
    which indicate we have to find a way to assembly those Z_i [seq_len * d_v] togather,where i is the index of header,
    make Z has the same dim of X [seq_len * d_model], the simplest way come to one's head is stack those Z_i togather. 
    d_v = d_model / num_head
    
    
    Put all the head togather, we got
    
    X [seq_len * d_model] * Head Weight W_k [num_head * d_model * d_k], resulted K_0 [num_head * seq_len * d_k]
    
"""
class SelfAttention(nn.Module):
    def  __init__(self, config):
        super().__init__()
        d_model = config.d_model
        seq_len = config.seq_len
        pass



    # input [bz * s_l * d_model]
    def forward(self, input):
        pass

