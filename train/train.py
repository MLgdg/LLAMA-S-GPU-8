import model
import config
import torch
import dataset
import time
from torch import nn 
import traceback
import json 
epoch = 100
CUDA=0
cuda_list = [0,1,2,3]
loss_base = 1
print_f = 20
cfg = json.loads('.conf/llama.json')
gpt = model.GPT2LMHeadModel(cfg).cuda(CUDA)
gpt = torch.nn.DataParallel(gpt, device_ids=cuda_list)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
traindata = torch.utils.data.DataLoader(dataset.TextData('./dongtai_v2tov4',cfg), batch_size=256,shuffle=True,num_workers=0,collate_fn=dataset.collate_fn,drop_last=True)
opt = torch.optim.Adam(gpt.parameters(), lr=1e-4, betas=(0.9, 0.99), eps=1e-10)


s1 = time.time()
for j in range(epoch):
    for i, data in enumerate(traindata):
        opt.zero_grad()
        s2 = time.time()
        nn.utils.clip_grad_norm_(gpt.parameters(), max_norm=5, norm_type=2)
        s3 = time.time()
        input_ids = data['input_ids'].cuda(CUDA)
        label_ids = data['label_ids'].cuda(CUDA)
        try:
            out, x = gpt(input_ids)
        except:
            traceback.print_exc()
            print(input_ids.shape,label_ids.shape)
            break
        s4 = time.time()
        #print(out.shape, label_ids.shape)
        loss = loss_fn(out.view(-1, out.size(-1)), label_ids.view(-1))
        #print(123)
        loss.backward()
        opt.step()
        s5 = time.time()
        if i % print_f==0:
            print('epoch:{}/{} batch:{}/{} data_time:{:.4f} clip_data:{:.4f}  model_data:{:.4f} back_data:{:.4f} loss:{:.6f}'.format(
                j,epoch,i,len(traindata),s2-s1, s3-s2,s4-s3, s5-s4, loss

            ))
            if loss < loss_base:
                torch.save(gpt.state_dict(), './gpt_model.pth')
                loss_base = loss
        s1 = s5
