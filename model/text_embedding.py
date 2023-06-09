import copy
import json
import logging
from io import open

import torch
from torch import nn
#from apex.normalization.fused_layer_norm import FusedLayerNorm
#from torch.nn import LayerNorm 
from model.trick import RMSNorm as LayerNorm
from model.mask import PadMasking, FutureMasking

class TextEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pmask = PadMasking(config.pad)
        self.tmask = FutureMasking()
    def forward(self, input_ids):
        #if token_type_ids is None:
        #    token_type_ids = torch.zeros_like(input_ids)
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                    ).unsqueeze(0).repeat(input_ids.size(0),1).to(input_ids.device)
        masks = self.pmask(input_ids) + self.tmask(input_ids)
        #text_causal_mask = self.tmask(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        #token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = (words_embeddings + position_embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return (embeddings, masks)
if __name__ == '__main__':
    import sys
    sys.path.append('../')
    config = json.load(open('../conf/uvat.json','r'))
    import utils
    config = utils.Config(config)
    mode = TextEmbeddings(config)
    a = torch.arange(0, 10).view(2,5).to(torch.long)
    b = torch.arange(0, 10).view(2,5).to(torch.long)
    print(mode(a,b).shape)