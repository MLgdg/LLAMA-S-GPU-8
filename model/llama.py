from model.transformer import Transformer, LLAMAHead
from model.gpus import GPUs
from model.text_embedding import TextEmbeddings
import torch.nn as nn
class LLAMA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.E = TextEmbeddings(config).cuda(config.gpulsit[0])
        print('model embedding done')
        #print(config.num_hidden_layers)
        T = Transformer(config.hidden_size, config.num_hidden_layers, config.num_attention_heads)
        self.T = GPUs(T, config.gpulsit)
        print('model Transformer done')
        H = LLAMAHead(self.E.word_embeddings.weight)
        H.set_embeddings_weights(self.E.word_embeddings.weight)
        print('model head done')
        self.H = H#.cuda(config.gpulsit[-1])
    def forward(self, token):

        out, pad_mask, causal_mask= self.E(token)
        #print("embed shape", out.shape)
        #out = out.permute(1, 0, 2)  # NLD -> LND
        out = self.T(out, pad_mask, causal_mask)
        #out = out.permute(1, 0, 2)

        out = self.H(out.cuda(self.config.gpulsit[0]))
        return out, out[:, -1, :]
