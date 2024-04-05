import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerConfig:
    """
    The config class is an easy way to parse those hyper params into model
    Since nowadays model architecture could be really deep, packing all the hypers into one config obj and pass this obj
    from one component to deeper component is more neat than every __init__ func have a bunch of param
    """

    def __init__(self,
                 d_model: int = 512,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 batch_size: int = 16,
                 seq_len: int = 256,
                 d_ff: int = 2048,
                 vocab_size=37000,
                 device="cuda"
                 ):
        # the main model size of the transformer model, in the whole model,  we will use d_model number vector to
        # represent meaning of the word (the location of this word in the word embedding space)
        self.d_model = d_model
        # number of the heads define when we do the attention operation, parallel, there will be [num_heads] heads using
        # the same inputs but different learnable params to the same operation, the concat res will be the final res of
        # attention operation
        self.num_heads = num_heads
        # the dropout layer is critical in deep learning model, dropout is fantastic technical that can proven the
        # model overfit. what the dropout layer doing is it "cover/cut" random a percentage of input when it is
        # running, so the model won't over-relay on a certain feature/path of the model. it will increase the
        # robustness of the model
        self.dropout = dropout
        self.batch_size = batch_size
        # seq_len is the max number of token the model could process in one operation, not like RNN the model process
        # the input token by token, all the attention computation in transformer could be done at teh same time, we have
        # to define the max number of token, so the model can create weight mat accordingly
        self.seq_len = seq_len
        # the inner layer dim of fully-connected feed-forward component
        self.d_ff = d_ff
        # the vocab size of the tokenized, will be used to generate embedding layer and final fully connected layer
        self.vocab_size = vocab_size
        # indicate where the whole model will be running, all the tensor involved in the computation need to be moved
        # on the same device
        self.device = device


def clone(component: nn.Module, num_of_copy: int) -> nn.ModuleList:
    """
    In the transformer structure, there will a lot of repeat component, for example, the identical layer of encoders and
    decoders. In order to create those identical components, we will need this clone function to create a list ModuleList
    :param component: the component will be copied
    :param num_of_copy: the number of copies will be in the final module list
    :return: a module list contain num_of_copy component
    """
    return nn.ModuleList([copy.deepcopy(component) for _ in range(num_of_copy)])


class MultiHeadAttention(nn.Module):
    """
    Multi Head attention is a foundation component of transformer model,
    What is does just repeat the scaled dot product attention operation several times parallel, each time we call it a
    Head
    """

    def __init__(self, config: TransformerConfig):

        super(MultiHeadAttention, self).__init__()
        # since those head are doing the attention operation at the same time, we better put them in a same matrix
        # to make it efficient. In that case, if we define single head dim as d_single_head
        # d_model = num_heads * d_single_head. before we do the scaled dot-product we have to assert, otherwise we can't
        # split the d_model evenly into heads
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        assert self.d_model // self.num_heads == 0, "the number of head need to be divided by d_model"
        # this linear nn.ModuleList contains the W_q,W_k,W_v,W_o. All of them have the same size the purpose the W_q,
        # W_k,W_v is for projection. to do the scale dot-production attention, we have to use query(q) * key(k) to
        # get score between q and k then use the score as weight to retrieve info from the v, but there is an issue,
        # the original input the attention is general. For example in self attention, the original input of attention
        # is the same, 3 identical matrix represent a general meaning of the sentence. to get a better result. We
        # want project the general meaning into a specific space (query space, key space and value space) and use
        # those projected(professional) value to do the scale dot-product this W_o is used when we concat and
        # aggregate each head's value into final attention res since those head might have the same result,
        # some may focus on less important relation between q and k, we need a learnable params to assign weight to
        # each head and their dim
        self.linears = clone(nn.Linear(self.d_model, self.d_model), 4)
        self.dropout = nn.Dropout(p=config.dropout)
        self.seq_len = config.seq_len

    def _scaled_dot_product(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                            mask: torch.Tensoe, dropout: nn.Dropout) -> torch.Tensor:
        """
        this function will actually do the scaled dot product mentioned in equation (1)
        for this function, all q k v need to be prepared, which it has already been split into head dim
        for the mask, it also should adjust to proper dim for broadcasting operation
        :param q: [batch_size * num_heads * seq_len * d_k]
        :param k: [batch_size * num_heads * seq_len * d_k]
        :param v: [batch_size * num_heads * seq_len * d_k]
        :param mask: [ batch_size * 1 * seq_len * 1]
        :param dropout: the dropout defined in the outer layer
        :return: the scaled dot-product result [batch_size * num_heads * seq_len * d_k]
        """
        score = (q @ k.transpose(-2, -1)) / torch.sqrt(self.d_k)

        if mask:
            # TODO explain the mask fill and why there is a small value
            score.masked_fill(mask == 0, 1e-9)
        score = F.softmax(score, dim=-1)
        # TODO explain why dropout before
        return dropout(score) @ v

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        this function implement the function in section 3.2.2
        :param q:[batch_size * seq_len * d_model]
        :param k:[batch_size * seq_len * d_model]
        :param v:[batch_size * seq_len * d_model]
        :param mask:[batch_size * seq_len]
        :return: result of multi head attention [batch_size * seq_len * d_model]
        """
        # get the batch since the q k v need to be reshaped latter. the number of sentence in the batch won't be all the
        # time the same, for example, the last batch may not be full
        batch_size = q.size(0)
        # TODO explain why have mask here

        # the mask is generated by tokenizer, usually the dim is [batch_size * seq_len] contains of 1 and 0
        # where 1 represent the position of the corresponding sentence is a meaningful token, otherwise it is a
        # padding. in order to use it, mask_fill the score, it has to meet the requirement of broadcasting with score
        # since the dim of score is [batch_size * num_heads * seq_len * d_k], the mask has to un-squeeze at dim 1 and
        # dim -1
        if mask:
            mask.unsqueeze(1)
            mask.unsqueeze(-1)
        # project the input q,k,v into according space to get actual query, key and value
        q_k_v = [w(mat) for mat, w in zip([q, k, v], self.linears)]
        # reshape the q , k  and v to into heads, constitute the multi head
        q_k_v = [mat.view(batch_size, self.seq_len, self.num_heads, -1) for mat in q_k_v]
        # transpose the number since the matmul only work on last two dim, to calculate the attention, we want to
        # compute q [...... seq_len * d_q] * k [..... d_k, seq_len]
        # after reshaping, the dim is [batch_size * seq_len, num_heads, d_k]
        # so dim 1 and dim 2 need to transpose
        q_k_v = [torch.transpose(mat, 1, 2) for mat in q_k_v]
        # after everything be prepared, the scaled dot-product will be conducted.
        # the output of that func is split into head, we need transpose the dim back and reshape the same dim
        # as the input, so in the transformer the following identical layer could keeping do the same attention
        # operation again and again

        # TODO why need contiguous()
        x = self._scaled_dot_product(*q_k_v).transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.linears[-1](x)


class FeedForward(nn.Module):
    """
    # todo what is purpose this feed forward layer
    """

    def __init__(self, config: TransformerConfig):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(config.d_model, config.d_ff)
        self.w_2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        this function implement the equation (2)
        :param x: [batch_size * seq_len * d_model]
        :return: [batch_size * seq_len * d_model]
        """
        # TODO explain the position of the dropout
        # according to the equation, this fully connected feed forward layer, this consists of two linear
        # transformations with a ReLU activation in between.
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embedding(nn.Module):
    """
    # todo
    """

    def __init__(self, config: TransformerConfig):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.d_model = torch.tensor(self.d_model).to(config.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #  todo explain  the sqrt in the embedding layer
        return self.embedding(x) / torch.sqrt(self.d_model)


class PositionalEmbedding(nn.Module):
    """
    Todo
    """

    def __init__(self, config):
        super(PositionalEmbedding, self).__init__()
        # todo
        self.pe = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :return:
        """
        batch_size = x.size(0)
        return x + self.pe[:batch_size]



class LayerNorm(nn.Module):





class Sublayer(nn.Module):



class EncoderLayer(nn.Module):


class Encoder(nn.Module):


class DecoderLayer(nn.Module):


class Decoder(nn.Module):

class Transformer(nn.Module)