
import model
import torch
import dataset
import time
from torch import nn 
import traceback
import json 
from model.text_embedding import TextEmbeddings
from model.attention import Attention
from model.transformer import ResidualAttentionBlock, FUCKHead
from dataset import dataset
from optimizer import opt
from conf.config import DictToClass
from fairscale.nn.pipe.balance import balance_by_time
import fairscale
epoch = 100
CUDA=0
loss_base = 100
print_f = 20
cfg = json.load(open('./conf/llama.json'))
cfg = DictToClass(cfg)


partitions = torch.cuda.device_count()                                                               
sample = torch.linspace(1,4096, 4096).view(2, 2048).to(torch.int)  
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
model = fairscale.nn.Pipe(model, balance=balance, chunks=8)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=cfg.pad)
traindata = torch.utils.data.DataLoader(dataset.TextData('./data/part-00499',cfg), batch_size=256,shuffle=True,num_workers=0,collate_fn=dataset.collate_fn,drop_last=True)


optimizer, scheduler = opt.opt(model.parameters())

s1 = time.time()
for j in range(epoch):
    for i, data in enumerate(traindata):
        optimizer.zero_grad()
        s2 = time.time()
        s3 = s2
        input_ids = data['input_ids'].cuda(cfg.gpulsit[0])
        label_ids = data['label_ids'].cuda(cfg.gpulsit[-1])
        try:
            out = model(input_ids)
        except:
            traceback.print_exc()
            print(input_ids.shape,label_ids.shape)
            break
        s4 = time.time()
        #print(out.shape, label_ids.shape)
        loss = loss_fn(out.view(-1, out.size(-1)), label_ids.view(-1))
        #print(123)
        loss.backward()
        nn.utils.clip_grad_norm_(llama.parameters(), max_norm=1, norm_type=2)
        
        opt.step()
        scheduler.step()
        s5 = time.time()
        if i % print_f==0:
            print('epoch:{}/{} batch:{}/{} data_time:{:.4f} clip_data:{:.4f}  model_data:{:.4f} back_data:{:.4f} loss:{:.6f}'.format(
                j,epoch,i,len(traindata),s2-s1, s3-s2,s4-s3, s5-s4, loss

            ))
            if loss < loss_base:
                torch.save(model.state_dict(), './fuck_model.pth')
                loss_base = loss
        s1 = s5
