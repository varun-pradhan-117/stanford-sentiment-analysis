'''
Code adapted from: https://medium.com/swlh/transformer-fine-tuning-for-sentiment-analysis-c000da034bb5
'''

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from typing import Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader
from tqdm import tqdm

from pytorch_transformers import BertTokenizer

n_cpu = multiprocessing.cpu_count()
MAX_LENGTH = 256

def read_sst5(data_dir, colnames=['label', 'text']):
    datasets = {}
    for t in ["train", "dev", "test"]:
        df = pd.read_csv(os.path.join(data_dir, f"sst_{t}.txt"), sep='\t', header=None, names=colnames)
        df['label'] = df['label'].astype(int)   # Categorical data type for truth labels
        df['label'] = df['label'] - 1  # Zero-index labels for PyTorch
        datasets[t] = df
    return datasets



class TextProcessor:
    # special tokens for classification and padding
    CLS = '[CLS]'
    PAD = '[PAD]'
    
    def __init__(self, tokenizer, label2id: dict,max_length: int=256):
        self.tokenizer=tokenizer
        self.label2id = label2id
        self.num_labels = len(label2id)
        self.CLS = tokenizer.vocab['[CLS]']
        self.PAD = tokenizer.vocab['[PAD]']
        self.max_length = max_length

    def encode(self,input):
        return list(self.tokenizer.convert_tokens_to_ids(token) for token in input)
     
    #Convert text (item[0]) to sequence of IDs and label (item[1]) to integer    
    def token2id(self, item: Tuple[str, str]):

        assert len(item) == 2   # Need a row of text AND labels
        label, text = item[0], item[1]
        assert isinstance(text, str)   # Need position 1 of input to be of type(str)
        inputs = self.tokenizer.tokenize(text)
        # Trim or pad dataset
        if len(inputs) >= self.max_length:
            inputs = inputs[:self.max_length - 1]
            ids = self.encode(inputs) + [self.CLS]
        else:
            pad = [self.PAD] * (self.max_length - len(inputs) - 1)
            ids = self.encode(inputs) + [self.CLS] + pad

        return np.array(ids, dtype='int64'), self.label2id[label]

    #Process each row
    def process_row(self, row):
        return self.token2id((row[1]['label'], row[1]['text']))

    def create_dataloader(self,
                          df: pd.DataFrame,
                          batch_size: int = 32,
                          shuffle: bool = False):
        tqdm.pandas()
        with ProcessPoolExecutor(max_workers=n_cpu) as executor:
            result = list(tqdm(executor.map(self.process_row, df.iterrows(), chunksize=8192),
                                            desc=f"Processing {len(df)} examples on {n_cpu} cores",
                                            total=len(df)))
        
        features = [r[0] for r in result]
        labels = [r[1] for r in result]

        dataset = TensorDataset(torch.tensor(features, dtype=torch.long),
                                torch.tensor(labels, dtype=torch.long))

        data_loader = DataLoader(dataset,
                                 batch_size=batch_size,
                                 num_workers=0,
                                 shuffle=shuffle,
                                 pin_memory=torch.cuda.is_available())

        return data_loader

if __name__ == "__main__":
    datasets = read_sst5('D:\\Github stuff\\Internship Stuff\\stanford-sentiment-analysis\\data')
    print(datasets['train'].head(10))
    labels = list(set(datasets["train"]['label'].tolist()))
    label2int = {label: i for i, label in enumerate(labels)}
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    CLS = tokenizer.vocab['[CLS]']  # classifier token
    PAD = tokenizer.vocab['[PAD]']  # pad token
    proc = TextProcessor(tokenizer, label2int, max_length=MAX_LENGTH)

    train_dl = proc.create_dataloader(datasets["train"], batch_size=32)
    valid_dl = proc.create_dataloader(datasets["dev"], batch_size=32)
    test_dl = proc.create_dataloader(datasets["test"], batch_size=32)

    print(len(train_dl), len(valid_dl), len(test_dl))