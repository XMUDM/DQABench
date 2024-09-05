# coding=utf-8
import os
import time
import string
import pandas as pd
import torch
from transformers import XLNetTokenizer
from prepare_data import xlnet_encode
import tqdm
from train import MyXLNetModel
from eval import get_metrics, eval_step
import argparse

def is_contain_chinese(sentence):
    sentence = ''.join(char for char in sentence if char not in string.punctuation)
    for char in sentence:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_en", default="ckpt/en_5class.tar", type=str,
                        help="The path of the English safe classifier")
    parser.add_argument("--model_zh", default="ckpt/zh_5class.tar", type=str,
                        help="The path of the Chinese safe classifier")
    parser.add_argument("--doc_maxlen", default="4000", type=int, help="The maximum input length")
    parser.add_argument("--segment_len", default="256", type=int, help="segment length")
    parser.add_argument("--overlap", default="50", type=int, help="Length of overlap between segments")
    parser.add_argument("--ngpu", default="1", type=int, help="The number of gpu")
    parser.add_argument("--pretrained_model_en", default="", type=str,
                        help="The directory of the English pretrained model, download from https://huggingface.co/xlnet/xlnet-base-cased")
    parser.add_argument("--pretrained_model_zh", default="", type=str,
                        help="The directory of the Chinese pretrained model, download from https://huggingface.co/hfl/chinese-xlnet-base")
    parser.add_argument("--test_set", default="zh_test.csv", type=str, help="The path of the test set")

    args = parser.parse_args()
    labels = ['general', 'gauss', 'tool', 'other', 'unsafe']
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for i, l in enumerate(labels)}
    num_classes = len(labels)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if (use_cuda and args.ngpu > 0) else "cpu")
    print('*' * 8, 'device:', device)
    en_xlnet = args.pretrained_model_en
    zh_xlnet = args.pretrained_model_zh
    tokenizer_en = XLNetTokenizer.from_pretrained(en_xlnet + 'spiece.model')
    tokenizer_zh = XLNetTokenizer.from_pretrained(zh_xlnet + 'spiece.model')
    model_en = MyXLNetModel(en_xlnet, num_classes, feature_extract=True, segment_len=args.segment_len).to(device)
    model_zh = MyXLNetModel(zh_xlnet, num_classes, feature_extract=True, segment_len=args.segment_len).to(device)
    pretrain_model = [model_en, model_zh]

    print('*' * 27, 'Loading model weights...')
    en_ckpt = torch.load(args.model_en)['net']
    zh_ckpt = torch.load(args.model_zh)['net']
    ckpts = [en_ckpt, zh_ckpt]
    for model, model_sd in zip(pretrain_model, ckpts):
        if device.type == 'cuda' and args.ngpu > 1:
            model.module.load_state_dict(model_sd)
        else:
            model.load_state_dict(model_sd)
    print('*' * 27, 'Model loaded success!')

    df = pd.read_csv(args.test_set, delimiter="\t", header=None)
    preds = []
    labels = []
    starttime = time.time()

    for index, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        label = row[0]
        question = row[1]
        try:
            if is_contain_chinese(question):
                print(index, "zh")
                tokenizer = tokenizer_zh
                model = model_zh
            else:
                print(index, "en")
                tokenizer = tokenizer_en
                model = model_en

        except Exception as e:
            print("error: ", e)
            print("index: ", index)
            print("question: ", question)
        x_data = xlnet_encode([question], tokenizer, args.doc_maxlen, print_time=False)
        inps = (x_data['input_ids'], x_data['token_type_ids'], x_data['attention_mask'])
        y_data = [label2id[label]]
        y_data = torch.tensor(y_data, dtype=torch.long)
        pred, labs = eval_step(model, inps, y_data, device)
        preds.append(pred)
        labels.append(labs)

    y_true = torch.cat(labels, dim=0)
    y_pred = torch.cat(preds, dim=0)
    endtime = time.time()
    print('evaluating costs: {:.2f}s'.format(endtime - starttime))
    get_metrics(y_true.cpu(), y_pred.cpu(), num_classes)