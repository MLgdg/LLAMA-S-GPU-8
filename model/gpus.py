
import torch 
import torch.nn as nn

from transformer import Transformer, LLAMAHead
from text_embedding import TextEmbeddings

GPU = [0,1,2,3,4,5,6,7]

class GPUs(nn.Module):
	def __init__(self, model, gpu_list):
		super().__init__()

	

