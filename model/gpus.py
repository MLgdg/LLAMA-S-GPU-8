
import torch 
import torch.nn as nn

from transformer import Transformer, LLAMAHead
from text_embedding import TextEmbeddings

GPU = [0,1,2,3,4,5,6,7]

class GPUs(nn.Module):
	def __init__(self, model, gpu_list):
		super().__init__()
		model = model.resblocks
		tmp_model = nn.ModuleList()
		self.model_gpu = []
		for i in gpu_list:
			for j in model[i*10: (i+1)+10]:
				tmp_model.append(j.cuda(i))
				self.model_gpu.append(i)
		self.model = nn.Sequential(*tmp_model)
	def forward(self, x, attn_mask=None):
		if attn_mask==None:
			for i in range(len(self.model_gpu)):
				x = self.model[i](x.cuda(self.model_gpu[i]))
		else:
			for i in range(len(self.model_gpu)):
				x = self.model[i](x.cuda(self.model_gpu[i]), attn_mask.cuda((self.model_gpu[i]))



			#self.model.append(model[i*10: (i+1)+10].)

		

