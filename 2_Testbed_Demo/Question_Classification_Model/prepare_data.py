# coding=utf-8
import time
import pandas as pd

import torch
from transformers import XLNetTokenizer

import sys
def read_data(filepath, tokenizer, label2id, maxlen):
    df_data = pd.read_csv(filepath, encoding='UTF-8', sep='\t', names=['label', 'content'], index_col=False)
    df_data = df_data.dropna()

    x_data, y_data = df_data['content'], df_data['label']
    print('*' * 27, x_data.shape, len(x_data[0]), y_data.shape)

    x_data = xlnet_encode(x_data, tokenizer, maxlen)

    y_data = [label2id[y] for y in y_data]
    y_data = torch.tensor(y_data, dtype=torch.long)

    return x_data, y_data



def xlnet_encode(texts, tokenizer, maxlen, print_time=True):
    starttime = time.time()
    if print_time:
        print('*'*27, 'start encoding...')
    inputs = tokenizer.batch_encode_plus(texts, return_tensors='pt', add_special_tokens=True,
                                         max_length=maxlen,
                                         padding='max_length',
                                         truncation='longest_first')

    endtime = time.time()
    if print_time:
        print('*'*27, 'data to ids finished...')
        print('*'*27, 'and it costs {} min {:.2f} s'.format(int((endtime-starttime)//60), (endtime-starttime)%60))
    return inputs



def load_data(filepath, tokenizer, batch_size, label2id, maxlen, shuffle=False):
    inputs, y_data = read_data(filepath, tokenizer, label2id, maxlen)

    inp_dset = torch.utils.data.TensorDataset(inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask'],
                                              y_data)
    inp_dloader = torch.utils.data.DataLoader(inp_dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=2)
    return inp_dloader
