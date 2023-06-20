import torch
from torch.utils.data import Dataset, DataLoader
import dataset.tokenization_glm as tokenization
import json 




"""
TODO
ChatGLM模型在训练和预测时需要将中文文本切分成固定长度的字符，下面是一些常用的中文文本切分方法：

基础版切分：将文本先分词，然后将每个词的长度限制在固定范围内，不足固定长度用特殊字符填充。这种方法比较简单，但容易导致信息损失。
增强版切分：在基础版切分的基础上，利用中文文本的语法和语义信息，将句子分割成更小的片段，使每个片段的长度都在固定范围内。例如，按照标点符号、虚词等将句子分割成若干个短语，然后根据短语的长度进行填充。
结合语言模型切分：使用预先训练好的语言模型（例如BERT等），对输入文本进行编码，然后根据编码后的向量信息进行切分。具体来说，可以将编码后的向量信息进行动态规划，找到一个最大得分的位置进行切分。
需要注意的是，不同的切分方法会对模型的性能产生不同的影响，需要根据具体情况选择合适的切分方法。此外，在切分时还需要考虑文本的语义和语法信息，避免过度切分或信息损失。
"""
def collate_fn(batch):
    input_ids = []
    label_ids = []
    l = []
    for i in batch:
        l.append(len(i['input_ids']))
        l.append(len(i['label_ids']))
    length = max(l)
    for i in batch:
        input_id = i['input_ids']
        label_id = i['label_ids']
        input_id = input_id + [3] * (length - len(input_id))
        label_id = label_id + [3] * (length - len(label_id))
        input_ids.append(input_id)
        label_ids.append(label_id)
    return {'input_ids': torch.tensor(input_ids), 'label_ids': torch.tensor(label_ids)}

def predata(tokenize, data, max_len = 2048):
    data = tokener.tokenize(data, add_dummy_prefix=False)                                           
    data = ["<sop>"] + data + ["eop"]
    if len(data) <= max_len:
        return [data]
    tmp = []
    for i in range(len(data) // max_len + 1):
        tmp.append(data[i * (max_len + 1):  (i + 1) * (max_len + 1)])
    return tmp

class TextData(Dataset):

    def __init__(self, path, config=None):
        super(TextData, self).__init__()
        #print(path)
        self.root = path
        self.config = config
        self.tokener = tokenization.SPTokenizer('./dataset/ice_text.model')
        self.data = []
        with open(path)as ff:
            for ll in ff:
                #print(ll)
                try:
                    data = json.loads(ll.strip()).get('content','')
                    # for line in data.split('。'):
                    #     if line:
                    data = predata(self.tokener, data, config.max_position_embeddings)
                    for i in data:
                        self.data.append(data)
                except:
                    print("get data error")
    
    def __getitem__(self, index):
        token = self.data[index]
        #b = self.tokener.tokenize(text,add_dummy_prefix=False) 
        token_id = self.tokener.convert_tokens_to_ids(token) 
        input_ids = token_id[:-1]
        label_ids = token_id[1:]
        return {'input_ids': input_ids, 'label_ids': label_ids}

    def __len__(self):
        return len(self.data)
        

                