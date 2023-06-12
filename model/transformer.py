import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from model.mask import PadMasking, FutureMasking
from model.trick import RMSNorm as LayerNorm
from model.trick import FeedForward
from model.text_embedding import TextEmbeddings
from model.attention import Attention
from conf.config import DictToClass
import fairscale

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
#TODO
"""
0、输入由原来的mask 改为seq_len,
1、输出序列的长度
2、在内部进行mask计算
3、修改mask实现代码
"""

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.attn = Attention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = FeedForward(d_model, 4*d_model)
        self.ln_2 = LayerNorm(d_model)
        #self.attn_mask = attn_mask

    def attention(self, x, mask=None):

        #self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, mask)

    def forward(self, data):
        x, mask = data
        x = x + self.attention(self.ln_1(x), mask) 
        x = x + self.mlp(self.ln_2(x))
        return (x, mask)

class Transformer(nn.Module):
    def __init__(self, width, layers, heads=64):
        super().__init__()
        self.width = width#8192
        self.layers = layers#80
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])
        
    def forward(self, x, mask=None):
        return self.resblocks(x, mask)

class FUCKHead(nn.Module):
    def __init__(self, model_embeddings_weights):
        super(FUCKHead, self).__init__()
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights
    def forward(self, data):
        x, mask = data
        lm_logits = self.decoder(x)
        return lm_logits




class LLAMA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.E = TextEmbeddings(config)
        print('model embedding done')
        #print(config.num_hidden_layers)
        self.T = Transformer(config.hidden_size, config.num_hidden_layers, config.num_attention_heads)
        #self.T = GPUs(T, config.gpulsit)
        print('model Transformer done')
        H = LLAMAHead(self.E.word_embeddings.weight)
        H.set_embeddings_weights(self.E.word_embeddings.weight)
        print('model head done')
        self.H = H#.cuda(config.gpulsit[-1])
    def forward(self, token):

        out, mask= self.E(token)
        #print("embed shape", out.shape)
        #out = out.permute(1, 0, 2)  # NLD -> LND
        out = self.T(out, mask)
        #out = out.permute(1, 0, 2)

        out = self.H(out)
if __name__=="__main__":
    cfg = json.load(open('./conf/llama.json'))
    cfg = DictToClass(cfg)
    llama = LLAMA(cfg)

