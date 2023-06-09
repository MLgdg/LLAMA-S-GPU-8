import copy
import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter



class Attention(nn.Module):
    def __init__(self, dim, heads, dropout=0, scale=True):
        super(Attention, self).__init__()
       
        assert dim % heads == 0
        #self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = heads
        self.split_size = dim
        self.scale = scale
        self.c_attn = nn.Linear(dim, dim*3)
        self.c_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    def _attn(self, q, k, v, mask):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        #nd, ns = w.size(-2), w.size(-1)
        #b = self.bias[:, :, ns-nd:ns, :ns]
        #print(w.shape)
        #print(mask.shape)
        mask = mask.unsqueeze(-3)
        #print(mask, w.shape)
        w += mask.type_as(w) #* w.new_tensor(-1e10)
        #print(w[0][0])
        #w = w * b - 1e10 * (1 - b)
        w = nn.Softmax(dim=-1)(w)
        w = self.dropout(w)
        #print(v.shape)
        return torch.matmul(w, v) #mask是

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, mask=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        a = self._attn(query, key, value, mask)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        
        return a