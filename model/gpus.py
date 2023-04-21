
import torch 
import torch.nn as nn

from transformer import Transformer, LLAMAHead
from text_embedding import TextEmbeddings

GPU = [0,1,2,3,4,5,6,7]


class LLAMA(nn.Module):
    def __init__(self, config, model_embeddings_weights)
        super().__init__()
        self.E = TextEmbeddings(config)
        self.T = Transformer(config.hidden_size, config.num_hidden_layers, config.num_attention_heads)
        self.H = LLAMAHead(self.E.word_embeddings.weight)
        self.H.set_embeddings_weights(self.E.word_embeddings.weight)
    def forward(self, token):
        out = self.E(token)
        out = self.T(out)
        out = self.H(out)
        return out, out[:, -1, :]