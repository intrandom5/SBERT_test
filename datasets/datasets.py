from torch.utils.data import Dataset
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from collections import defaultdict
from typing import List, Tuple
import random


class KorSTSDatasets(Dataset):
    def __init__(self, dir_x, dir_y):
        self.x = np.load(dir_x, allow_pickle=True)
        self.y = np.load(dir_y, allow_pickle=True)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sentence1, sentence2 = self.x[idx]
        data = torch.IntTensor(sentence1), torch.IntTensor(sentence2)
        # cosine similarity의 범위 [-1. ~ 1.] 사이 값으로 정규화 필요.
        # label = float(self.y[idx]) * 0.4 - 1
        
        label = float(self.y[idx])
        return data, label

def KorSTS_collate_fn(batch):
    # batch = list([((s1, s2), label), ((s1, s2), label), ...])
    s1_batches = []
    s2_batches = []
    labels = []
    for b in batch:
        data, label = b
        s1, s2 = data
        s1_batches.append(s1)
        s2_batches.append(s2)
        labels.append(label)
        
    s1_batch = pad_sequence(s1_batches, batch_first=True, padding_value=0)
    s2_batch = pad_sequence(s2_batches, batch_first=True, padding_value=0)
    return s1_batch.long(), s2_batch.long(), torch.FloatTensor(labels)

def bucket_pair_indices(
    sentence_length: List[Tuple[int, int]],
    batch_size: int,
    max_pad_len: int
) -> List[List[int]]:
    batch_indices_list = []
    bucket = defaultdict(list)
    for idx, length in enumerate(sentence_length):
        s1_len, s2_len = length
        x = s1_len//max_pad_len
        y = s2_len//max_pad_len
        bucket[(x, y)].append(idx)
        if len(bucket[(x, y)]) == batch_size:
            batch_indices_list.append(bucket[(x, y)])
            bucket[(x, y)] = []
    for key in bucket.keys():
        batch_indices_list.append(bucket[key])

    random.shuffle(batch_indices_list)

    return batch_indices_list

