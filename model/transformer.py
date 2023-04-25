import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from model.mask import PadMasking, FutureMasking
from model.trick import RMSNorm as LayerNorm
from model.trick import FeedForward
from model.text_embedding import TextEmbeddings
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

# class FeedForward(nn.Module):

#     def __init__(self, d_model):
#         super().__init__()
#         self.l1 = nn.Linear(d_model, d_model * 4)
#         self.ac = QuickGELU()
#         self.l2 = nn.Linear(d_model * 4, d_model)

#     def forward(self, x):
#         return self.l2(self.ac(self.l1(x)))

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = FeedForward(d_model, 4*d_model)
        self.ln_2 = LayerNorm(d_model)
        #self.attn_mask = attn_mask
    def attention(self, x, attn_mask=None):

        #self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x, attn_mask=None):
        x = x + self.attention(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width, layers, heads=64):
        super().__init__()
        self.width = width#8192
        self.layers = layers#80
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])
        
    def forward(self, x, attn_mask=None):
        return self.resblocks(x, attn_mask=None)

class LLAMAHead(nn.Module):
    def __init__(self, model_embeddings_weights):
        super(LLAMAHead, self).__init__()
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights
    def forward(self, x):
        lm_logits = self.decoder(hidden_state)
        return lm_logits

class LLAMA(nn.Module):
    def __init__(self, config, model_embeddings_weights):
        super().__init__()
        self.E = TextEmbeddings(config)
        self.T = Transformer(config.hidden_size, config.num_hidden_layers, config.num_attention_heads)
        self.H = LLAMAHead(self.E.word_embeddings.weight)
        self.H.set_embeddings_weights(self.E.word_embeddings.weight)
    def forward(self, token):
        out = self.E(token)
        out = out.permute(1, 0, 2)  # NLD -> LND
        out = self.T(out)
        out = out.permute(1, 0, 2)
        out = self.H(out)
        return out, out[:, -1, :]




