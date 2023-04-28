
import torch 
import torch.nn as nn

from model.transformer import Transformer, LLAMAHead
from model.text_embedding import TextEmbeddings

GPU = [0,1,2,3]

class GPUs(nn.Module):
	def __init__(self, model, gpu_list):
		super().__init__()
		model = model.resblocks
		tmp_model = nn.ModuleList()
		self.model_gpu = []
		for i in gpu_list:
			for j in model[i*15: (i+1)*15]:
				tmp_model.append(j.cuda(i))
				self.model_gpu.append(i)
		self.model = nn.Sequential(*tmp_model)
	def forward(self, x, pad_mask=None, causal_mask=None):
		if attn_mask==None:
			for i in range(len(self.model_gpu)):
				x = self.model[i](x.cuda(self.model_gpu[i]))
		else:
			for i in range(len(self.model_gpu)):
				x = self.model[i](x.cuda(self.model_gpu[i]), pad_mask.cuda(self.model_gpu[i]), causal_mask.cuda(self.model_gpu[i]))

		return x

			#self.model.append(model[i*10: (i+1)+10].)

		

