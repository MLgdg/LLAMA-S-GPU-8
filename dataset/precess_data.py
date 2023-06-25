import torch
from torch.utils.data import Dataset, DataLoader
import dataset.tokenization_glm as tokenization
import json 

import sys 

def predata(tokenize, data, max_len = 2048):
	data = tokener.tokenize(data, add_dummy_prefix=False)                                           
	data = ["<sop>"] + data + ["eop"]
	if len(data) <= max_len:
	    return [data]
	tmp = []
	for i in range(len(data) // max_len + 1):
	    tmp.append(data[i * (max_len + 1):  (i + 1) * (max_len + 1)])
	return tmp

tokener = tokenization.SPTokenizer('./dataset/ice_text.model')


name = sys.argv[1]
root = "/root/paddlejob/workspace/gaoqingdong/learning/fuck/LLAMA-S-GPU-8/data/"
save_path = "/root/paddlejob/workspace/gaoqingdong/learning/fuck/LLAMA-S-GPU-8/data/"
if __name__ == '__main__':
	w = open('{}{}_token_id'.format(save_path, name),'w')
	with open(root+name)as ff:
		for ll in ff:
			data = json.loads(ll.strip()).get('content','')
			tokens = predata(tokener, data)
			for token in tokens:
				w.write("{}\n".format(json.dumps({"token_id": token})))
	w.clsoe()


	