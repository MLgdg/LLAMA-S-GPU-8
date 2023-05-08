import torch
from torch.utils.data import Dataset, DataLoader
import dataset.tokenization as tokenization
import json 

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
        input_id = input_id + [0] * (length - len(input_id))
        label_id = label_id + [0] * (length - len(label_id))
        input_ids.append(input_id)
        label_ids.append(label_id)
    return {'input_ids': torch.tensor(input_ids), 'label_ids': torch.tensor(label_ids)}
        
class TextData(Dataset):

    def __init__(self, path, config=None):
        super(TextData, self).__init__()
        #print(path)
        self.root = path
        self.config = config
        self.data = []
        with open(path)as ff:
            for ll in ff:
                #print(ll)
                try:
                    data = json.loads(ll.strip()).get('content','')
                    for line in data.split('。'):
                        if line:
                            self.data.append(data)
                except:
                    pass
        self.tokener = tokenization.BertTokenizer('./dataset/vocab.txt')
    def __getitem__(self, index):
        data = self.tokener.encode(self.data[index])[:(self.config.max_position_embeddings-2)]
        data = [101] + data + [105]
        input_ids = data[:-1]
        label_ids = data[1:]
        return {'input_ids': input_ids, 'label_ids': label_ids}

    def __len__(self):
        return len(self.data)
        

                