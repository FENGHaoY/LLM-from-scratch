import torch
from torch.utils.data import Dataset, DataLoader
#A:对文本进行编码
#B:创建input -> target对
#C:返回数据集长度和指定行
class GPTDatasetv1(Dataset):
    def __init__(self, text, tokenizer, context_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(text)
        for i in range(0, len(token_ids)-context_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i : i + context_length]))
            self.target_ids.append(torch.tensor(token_ids[i+1 : i+1+context_length]))
    
    def __len__(self):
        return(len(self.input_ids))
    
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]

#根据数据集建立Dataloader

def create_dataloader(text, tokenizer, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    dataset = GPTDatasetv1(text, tokenizer, context_length=max_length,stride=stride)
    dataloader = DataLoader(dataset=dataset, 
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last,
                            num_workers=num_workers)
    
    return dataloader