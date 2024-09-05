# coding=utf-8
import argparse
import os
import torch
from transformers import XLNetTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
            confusion_matrix, classification_report
import time
from prepare_data import load_data
from train import MyXLNetModel, printbar


@torch.no_grad()
def eval_step(model, inps, labs, device):
    input_ids, token_type_ids, attention_mask = inps
    input_ids = input_ids.to(device)
    token_type_ids = token_type_ids.to(device)
    attention_mask = attention_mask.to(device)
    labs = labs.to(device)

    model.eval()

    # forward
    logits = model(input_ids, token_type_ids, attention_mask)
    pred = torch.argmax(logits, dim=-1)

    return pred, labs


def evaluate(model, test_dloader, device):
    starttime = time.time()
    print('*' * 27, 'start evaluating...')
    printbar()
    preds, labels = [], []
    for step, (inp_ids, type_ids, att_mask, labs) in enumerate(test_dloader, start=1):
        inps = (inp_ids, type_ids, att_mask)
        pred, labs = eval_step(model, inps, labs, device)
        preds.append(pred)
        labels.append(labs)

    y_true = torch.cat(labels, dim=0)
    y_pred = torch.cat(preds, dim=0)
    endtime = time.time()
    print('evaluating costs: {:.2f}s'.format(endtime - starttime))
    return y_true.cpu(), y_pred.cpu()


def get_metrics(y_true, y_pred, num_classes):
    if num_classes == 2:
        print('*'*27, 'precision_score:', precision_score(y_true, y_pred, pos_label=1))
        print('*'*27, 'recall_score:', recall_score(y_true, y_pred, pos_label=1))
        print('*'*27, 'f1_score:', f1_score(y_true, y_pred, pos_label=1))
    else:
        average = 'weighted'
        print('*'*27, average+'_precision_score:{:.3f}'.format(precision_score(y_true, y_pred, average=average)))
        print('*'*27, average+'_recall_score:{:.3}'.format(recall_score(y_true, y_pred, average=average)))
        print('*'*27, average+'_f1_score:{:.3f}'.format(f1_score(y_true, y_pred, average=average)))

    print('*'*27, 'accuracy:{:.3f}'.format(accuracy_score(y_true, y_pred)))
    print('*'*27, 'confusion_matrix:\n', confusion_matrix(y_true, y_pred))
    print('*'*27, 'classification_report:\n', classification_report(y_true, y_pred))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", default="", type=str,
                        help="The directory of the pretrained model, download from https://huggingface.co/hfl/chinese-xlnet-base or https://huggingface.co/xlnet/xlnet-base-cased")
    parser.add_argument("--test_dataset", default="", type=str, help="The path of the test_dataset")
    parser.add_argument("--save_dir", default="", type=str, help="The path of the save_dir")
    parser.add_argument("--last_new_checkpoint", default="", type=str, help="The path of the last_new_checkpoint")
    parser.add_argument("--labels", default=['unsafe','other'], type=list, help="The list of labels")
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--doc_maxlen", default="4000", type=int, help="The maximum input length")
    parser.add_argument("--segment_len", default="256", type=int, help="segment length")
    parser.add_argument("--ngpu", default="1", type=int, help="The number of gpu")
    parser.add_argument("--feature_extract", default=True, type=bool, help="Whether xlnet is only used as a feature extractor, if false, then xlnet is also involved in training for fine-tuning")

    args = parser.parse_args()
    tokenizer = XLNetTokenizer.from_pretrained(args.pretrained_model + 'spiece.model')

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if (use_cuda and args.ngpu > 0) else "cpu")
    print('*' * 8, 'device:', device)
    label2id = {l: i for i, l in enumerate(args.labels)}
    id2label = {i: l for i, l in enumerate(args.labels)}
    num_classes = len(args.labels)
    checkpoint = args.save_dir + args.last_new_checkpoint
    test_dloader = load_data(args.test_dataset, tokenizer, args.batch_size, label2id, args.doc_maxlen)

    sample_batch = next(iter(test_dloader))
    print('*' * 27, 'sample_batch:', len(sample_batch), sample_batch[0].size(), sample_batch[0].dtype,
          sample_batch[1].size(), sample_batch[1].dtype,
          sample_batch[2].size(), sample_batch[2].dtype,
          sample_batch[3].size(), sample_batch[3].dtype)

    model = MyXLNetModel(args.pretrained_model, num_classes, args.feature_extract, segment_len=args.segment_len)
    model = model.to(device)
    if args.ngpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))

    print('*' * 27, 'Loading model weights...')
    ckpt = torch.load(checkpoint)
    model_sd = ckpt['net']
    if device.type == 'cuda' and args.ngpu > 1:
        model.module.load_state_dict(model_sd)
    else:
        model.load_state_dict(model_sd)
    print('*' * 27, 'Model loaded success!')

    y_true, y_pred = evaluate(model, test_dloader, device)
    get_metrics(y_true, y_pred, num_classes)
