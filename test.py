import numpy as np
import json
import torch
import torch.nn.functional as F
from torch import nn
from model.mask import PadMasking, FutureMasking
from model.trick import RMSNorm as LayerNorm
from model.trick import FeedForward
from model.text_embedding import TextEmbeddings
from model.attention import Attention
from model.transformer import ResidualAttentionBlock, FUCKHead
from conf.config import DictToClass
import fairscale
from fairscale.nn.pipe.balance import balance_by_time

partitions = torch.cuda.device_count()                                                               
sample = torch.linspace(1,1024, 1024).view(2,512).to(torch.int)  

#balance = balance_by_time(partitions, model, sample)                                                 
# model = Pipe(model, balance)  

cfg = json.load(open('./conf/llama.json'))
cfg = DictToClass(cfg)
layers = cfg.num_hidden_layers
width = cfg.hidden_size
heads = cfg.num_attention_heads

emb = TextEmbeddings(cfg)
tra = [ResidualAttentionBlock(width, heads) for _ in range(layers)]
head = FUCKHead(emb.word_embeddings.weight)
model = [emb]+tra+[head]
model = torch.nn.Sequential(*model)
balance = balance_by_time(partitions, model, sample)  
print(balance)
model = fairscale.nn.Pipe(model, balance=balance, chunks=2)

x = torch.linspace(1,1024, 1024).view(2,512).to(torch.int).cuda(0)
a, b = model(x)
print(a)