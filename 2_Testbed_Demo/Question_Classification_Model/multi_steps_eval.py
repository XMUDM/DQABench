# coding=utf-8
import os
import time
import string
import pandas as pd
import torch
from transformers import XLNetTokenizer, XLNetModel, XLNetConfig
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix, classification_report
from prepare_data import xlnet_encode
import tqdm
from train import MyXLNetModel
from eval import eval_step, get_metrics
import argparse


def is_contain_chinese(sentence):
    sentence = ''.join(char for char in sentence if char not in string.punctuation)
    for char in sentence:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--safety_model_en", default="ckpt/safety_en_ckpt.tar", type=str,
                        help="The path of the English safe classifier")
    parser.add_argument("--safety_model_zh", default="ckpt/safety_zh_ckpt.tar", type=str,
                        help="The path of the Chinese safe classifier")
    parser.add_argument("--db_model_en", default="ckpt/db_en_multi_ckpt.tar", type=str,
                        help="The path of the English db classifier")
    parser.add_argument("--db_model_zh", default="ckpt/db_zh_multi_ckpt.tar", type=str,
                        help="The path of the Chinese db classifier")
    parser.add_argument("--doc_maxlen", default="4000", type=int, help="The maximum input length")
    parser.add_argument("--segment_len", default="256", type=int, help="segment length")
    parser.add_argument("--ngpu", default="1", type=int, help="The number of gpu")
    parser.add_argument("--pretrained_model_en", default="", type=str,
                        help="The directory of the English pretrained model, download from https://huggingface.co/xlnet/xlnet-base-cased")
    parser.add_argument("--pretrained_model_zh", default="", type=str,
                        help="The directory of the Chinese pretrained model, download from https://huggingface.co/hfl/chinese-xlnet-base")
    parser.add_argument("--test_set", default="zh_test.csv", type=str, help="The path of the test set")

    args = parser.parse_args()
    safety_labels = ['unsafe', 'other']
    safety_label2id = {l: i for i, l in enumerate(safety_labels)}
    safety_id2label = {i: l for i, l in enumerate(safety_labels)}
    safety_num_classes = len(safety_labels)
    db_labels = ['general', 'gauss', 'tool', 'other']
    db_label2id = {l: i for i, l in enumerate(db_labels)}
    db_id2label = {i: l for i, l in enumerate(db_labels)}
    db_num_classes = len(db_labels)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if (use_cuda and args.ngpu > 0) else "cpu")
    print('*' * 8, 'device:', device)
    en_xlnet = args.pretrained_model_en
    zh_xlnet = args.pretrained_model_zh
    tokenizer_en = XLNetTokenizer.from_pretrained(en_xlnet + 'spiece.model')
    tokenizer_zh = XLNetTokenizer.from_pretrained(zh_xlnet + 'spiece.model')
    safety_model_en = MyXLNetModel(en_xlnet, safety_num_classes, feature_extract=True, segment_len=args.segment_len).to(device)
    safety_model_zh = MyXLNetModel(zh_xlnet, safety_num_classes, feature_extract=True, segment_len=args.segment_len).to(device)
    db_model_en = MyXLNetModel(en_xlnet, db_num_classes, feature_extract=True, segment_len=args.segment_len).to(device)
    db_model_zh = MyXLNetModel(zh_xlnet, db_num_classes, feature_extract=True, segment_len=args.segment_len).to(device)
    pretrain_model = [safety_model_en, safety_model_zh, db_model_en, db_model_zh]

    print('*' * 27, 'Loading model weights...')
    safety_en_ckpt = torch.load(args.safety_model_en)['net']
    safety_zh_ckpt = torch.load(args.safety_model_zh)['net']
    db_zh_ckpt = torch.load(args.db_model_zh)['net']
    db_en_ckpt = torch.load(args.db_model_en)['net']
    ckpts = [safety_en_ckpt, safety_zh_ckpt, db_en_ckpt, db_zh_ckpt]
    for model, model_sd in zip(pretrain_model, ckpts):
        if device.type == 'cuda' and args.ngpu > 1:
            model.module.load_state_dict(model_sd)
        else:
            model.load_state_dict(model_sd)
    print('*' * 27, 'Model loaded success!')

    df = pd.read_csv(args.test_set, delimiter="\t", header=None)
    safety_preds = []
    safety_labels = []
    db_preds = []
    db_labels = []
    starttime = time.time()
    for index, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        label = row[0]
        question = row[1]
        # print(langid.classify(question))
        if is_contain_chinese(question):
            print(index, "zh")
            tokenizer = tokenizer_zh
            safety_model = safety_model_zh
            db_model = db_model_zh
        else:
            print(index, "en")
            tokenizer = tokenizer_en
            safety_model = safety_model_en
            db_model = db_model_en

        x_data = xlnet_encode([question], tokenizer, args.doc_maxlen, print_time=False)
        inps = (x_data['input_ids'], x_data['token_type_ids'], x_data['attention_mask'])
        new_label = label if label == "unsafe" else "other"
        safety_y_data = [safety_label2id[new_label]]
        safety_y_data = torch.tensor(safety_y_data, dtype=torch.long)
        pred, labs = eval_step(safety_model, inps, safety_y_data, device)
        safety_preds.append(pred)
        safety_labels.append(labs)

        if pred.tolist()[0] == 1:
            new_label = label if label != "unsafe" else "other"
            db_y_data = [db_label2id[new_label]]
            db_y_data = torch.tensor(db_y_data, dtype=torch.long)
            pred, labs = eval_step(db_model, inps, db_y_data, device)
            db_preds.append(pred)
            db_labels.append(labs)

    safety_y_true = torch.cat(safety_labels, dim=0)
    safety_y_pred = torch.cat(safety_preds, dim=0)
    db_y_true = torch.cat(db_labels, dim=0)
    db_y_pred = torch.cat(db_preds, dim=0)
    endtime = time.time()
    print('evaluating costs: {:.2f}s'.format(endtime - starttime))
    get_metrics(safety_y_true.cpu(), safety_y_pred.cpu(), safety_num_classes)
    get_metrics(db_y_true.cpu(), db_y_pred.cpu(), db_num_classes)

