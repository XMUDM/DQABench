# coding=utf-8
import argparse
import os
import torch
import torch.nn.functional as F
from transformers import XLNetTokenizer, XLNetModel, XLNetConfig

from matplotlib import pyplot as plt
import copy
import datetime
import pandas as pd
from sklearn.metrics import accuracy_score
import time
import sys

from prepare_data import load_data


def printbar():
    nowtime = datetime.datetime.now().strftime('%Y-%m_%d %H:%M:%S')
    print('\n' + "=========="*8 + '%s'%nowtime)


class NyAttentioin(torch.nn.Module):
    def __init__(self, hidden_size, attensize_size):
        super(NyAttentioin, self).__init__()

        self.attn = torch.nn.Linear(hidden_size, attensize_size)
        self.ctx = torch.nn.Linear(attensize_size, 1, bias=False)

    # inputs: [b, seq_len, hidden_size]
    def forward(self, inputs):
        u = self.attn(inputs).tanh() # [b, seq_len, hidden_size]=>[b, seq_len, attention_size]
        scores = self.ctx(u) # [b, seq_len, attention_size]=>[b, seq_len, 1]
        attn_weights = F.softmax(scores, dim=1) # [b, seq_len, 1]

        out = torch.bmm(inputs.transpose(1, 2), attn_weights) # [b, seq_len, hidden_size]=>[b, hidden_size, seq_len]x[b, seq_len, 1]=>[b, hidden_size, 1]
        return torch.squeeze(out, dim=-1) # [b, hidden_size, 1]=>[b, hidden_size]


class MyXLNetModel(torch.nn.Module):
    def __init__(self, pretrained_model_dir, num_classes, feature_extract, segment_len=150, dropout_p=0.5):
        super(MyXLNetModel, self).__init__()

        self.seg_len = segment_len

        self.config = XLNetConfig.from_json_file(pretrained_model_dir + 'config.json')
        self.config.mem_len = 150  # enable the memory #
        self.xlnet = XLNetModel.from_pretrained(pretrained_model_dir, config=self.config)

        if feature_extract:
            for p in self.xlnet.parameters():
                p.requires_grad = False

        d_model = self.config.hidden_size  # 768
        self.attention_layer1 = NyAttentioin(d_model, d_model // 2)
        self.attention_layer2 = NyAttentioin(d_model, d_model // 2)

        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.fc = torch.nn.Linear(d_model, num_classes)


    def get_segments_from_one_batch(self, input_ids, token_type_ids, attention_mask):
        doc_len = input_ids.shape[1]
        q, r = divmod(doc_len, self.seg_len)
        if r > 0:
            split_chunks = [self.seg_len] * q + [r]
        else:  # r==0
            split_chunks = [self.seg_len] * q

        input_ids = torch.split(input_ids, split_size_or_sections=split_chunks, dim=1)
        token_type_ids = torch.split(token_type_ids, split_size_or_sections=split_chunks, dim=1)
        attention_mask = torch.split(attention_mask, split_size_or_sections=split_chunks, dim=1)

        split_inputs = [{'input_ids': input_ids[seg_i], 'token_type_ids': token_type_ids[seg_i],
                         'attention_mask': attention_mask[seg_i]}
                        for seg_i in range(len(input_ids))]
        return split_inputs


    def forward(self, input_ids, token_type_ids, attention_mask):
        split_inputs = self.get_segments_from_one_batch(input_ids, token_type_ids, attention_mask)

        lower_intra_seg_repr = []
        mems = None
        for idx, seg_inp in enumerate(split_inputs):
            #last_hidden, mems = self.xlnet(**seg_inp, mems=mems)  # last_hidden: [b, 150, 768] mems: list: [150, b, 768]
            output = self.xlnet(**seg_inp, mems=mems, use_mems=True)
            last_hidden, mems = output["last_hidden_state"], output["mems"]
            lower_intra_seg_repr.append(self.attention_layer1(last_hidden))  # attention_layer1: [b, 150, 768]=>[b, 768]

        lower_intra_seg_repr = [torch.unsqueeze(seg, dim=1) for seg in lower_intra_seg_repr]  # list: [b, 768]=>[b, 1, 768]
        lower_intra_seg_repr = torch.cat(lower_intra_seg_repr, dim=1)  # [b, 1, 768]=>[b, num_seg, 768]

        higher_inter_seg_repr = self.attention_layer2(lower_intra_seg_repr)

        logits = self.fc(self.dropout(higher_inter_seg_repr))  # [b, 768]=>[b, num_classes]
        return logits



def train_step(model, inps, labs, optimizer):
    input_ids, token_type_ids, attention_mask = inps
    input_ids = input_ids.to(device)
    token_type_ids = token_type_ids.to(device)
    attention_mask = attention_mask.to(device)
    labs = labs.to(device)

    model.train()
    optimizer.zero_grad()

    # forward
    logits = model(input_ids, token_type_ids, attention_mask)
    loss = loss_func(logits, labs)

    pred = torch.argmax(logits, dim=-1)
    metric = metric_func(pred.cpu().numpy(), labs.cpu().numpy())

    # backward
    loss.backward()
    optimizer.step()

    return loss.item(), metric.item()


def validate_step(model, inps, labs):
    input_ids, token_type_ids, attention_mask = inps
    input_ids = input_ids.to(device)
    token_type_ids = token_type_ids.to(device)
    attention_mask = attention_mask.to(device)
    labs = labs.to(device)

    model.eval()

    # forward
    with torch.no_grad():
        logits = model(input_ids, token_type_ids, attention_mask)
        loss = loss_func(logits, labs)

        pred = torch.argmax(logits, dim=-1)
        metric = metric_func(pred.cpu().numpy(), labs.cpu().numpy())

    return loss.item(), metric.item()


def train_model(model, train_dloader, val_dloader, save_dir, ngpu, optimizer, scheduler_1r=None, init_epoch=0, num_epochs=10, print_every=150):
    starttime = time.time()
    print('*' * 27, 'start training...')
    printbar()

    best_metric = 0.
    for epoch in range(init_epoch+1, init_epoch+num_epochs+1):
        loss_sum, metric_sum = 0., 0.
        for step, (inp_ids, type_ids, att_mask, labs) in enumerate(train_dloader, start=1):
            inps = (inp_ids, type_ids, att_mask)
            loss, metric = train_step(model, inps, labs, optimizer)
            loss_sum += loss
            metric_sum += metric

            if step % print_every == 0:
                print('*'*27, f'[step = {step}] loss: {loss_sum/step:.3f}, {metric_name}: {metric_sum/step:.3f}')

        val_loss_sum, val_metric_sum = 0., 0.
        for val_step, (inp_ids, type_ids, att_mask, labs) in enumerate(val_dloader, start=1):
            inps = (inp_ids, type_ids, att_mask)
            val_loss, val_metric = validate_step(model, inps, labs)
            val_loss_sum += val_loss
            val_metric_sum += val_metric

        if scheduler_1r:
            scheduler_1r.step()

        # columns=['epoch', 'loss', metric_name, 'val_loss', 'val_'+metric_name]
        record = (epoch, loss_sum/step, metric_sum/step, val_loss_sum/val_step, val_metric_sum/val_step)
        df_history.loc[epoch - 1] = record

        print('EPOCH = {} loss: {:.3f}, {}: {:.3f}, val_loss: {:.3f}, val_{}: {:.3f}'.format(
               record[0], record[1], metric_name, record[2], record[3], metric_name, record[4]))
        printbar()

        current_metric_avg = val_metric_sum/val_step
        if current_metric_avg > best_metric:
            best_metric = current_metric_avg
            checkpoint = save_dir + f'epoch{epoch:03d}_valacc{current_metric_avg:.3f}_ckpt.tar'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if device.type == 'cuda' and ngpu > 1:
                model_sd = copy.deepcopy(model.module.state_dict())
            else:
                model_sd = copy.deepcopy(model.state_dict())
            torch.save({
                'loss': loss_sum / step,
                'epoch': epoch,
                'net': model_sd,
                'opt': optimizer.state_dict(),
            }, checkpoint)


    endtime = time.time()
    time_elapsed = endtime - starttime
    print('*' * 27, 'training finished...')
    print('*' * 27, 'and it costs {} h {} min {:.2f} s'.format(int(time_elapsed // 3600),
                                                               int((time_elapsed % 3600) // 60),
                                                               (time_elapsed % 3600) % 60))

    print('Best val Acc: {:4f}'.format(best_metric))
    return df_history


def plot_metric(df_history, metric, imgs_dir):
    plt.figure()

    train_metrics = df_history[metric]
    val_metrics = df_history['val_' + metric]

    epochs = range(1, len(train_metrics) + 1)

    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')

    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])
    if not os.path.exists(imgs_dir):
        os.makedirs(imgs_dir)
    plt.savefig(imgs_dir + 'xlnet_'+ metric + '.png')
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_model", default="", type=str,
                        help="The directory of the pretrained model, download from https://huggingface.co/hfl/chinese-xlnet-base or https://huggingface.co/xlnet/xlnet-base-cased")
    parser.add_argument("--train_dataset", default="", type=str, help="The path of the train_dataset")
    parser.add_argument("--val_dataset", default="", type=str, help="The path of the val_dataset")
    parser.add_argument("--save_dir", default="", type=str, help="The path of the save_dir")
    parser.add_argument("--imgs_dir", default="", type=str, help="The path of the imgs_dir")
    parser.add_argument("--last_new_checkpoint", default="", type=str, help="The path of the last_new_checkpoint")
    parser.add_argument("--labels", default=['unsafe','other'], type=list, help="The list of labels")
    parser.add_argument("--LR", default=5e-4, type=float, help="learning rate")
    parser.add_argument("--EPOCHS", default=1, type=int, help="The number of epochs")
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--doc_maxlen", default="4000", type=int, help="The maximum input length")
    parser.add_argument("--segment_len", default="256", type=int, help="segment length")
    parser.add_argument("--overlap", default="50", type=int, help="Length of overlap between segments")
    parser.add_argument("--ngpu", default="1", type=int, help="The number of gpu")
    parser.add_argument("--feature_extract", default=True, type=bool, help="Whether xlnet is only used as a feature extractor, if false, then xlnet is also involved in training for fine-tuning")
    parser.add_argument("--train_from_scrach", default=True, type=bool, help="Whether to start training the model all over again")

    args = parser.parse_args()
    tokenizer = XLNetTokenizer.from_pretrained(args.pretrained_model + 'spiece.model')

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if (use_cuda and args.ngpu > 0) else "cpu")
    print('*' * 8, 'device:', device)
    label2id = {l: i for i, l in enumerate(args.labels)}
    id2label = {i: l for i, l in enumerate(args.labels)}
    num_classes = len(args.labels)
    loss_func = torch.nn.CrossEntropyLoss()
    metric_func = lambda y_pred, y_true: accuracy_score(y_true, y_pred)
    metric_name = 'acc'
    df_history = pd.DataFrame(columns=["epoch", "loss", metric_name, "val_loss", "val_" + metric_name])

    train_dloader = load_data(args.train_dataset, tokenizer, args.batch_size, label2id, args.doc_maxlen, shuffle=True)
    val_dloader = load_data(args.val_dataset, tokenizer, args.batch_size, label2id, args.doc_maxlen)

    sample_batch = next(iter(train_dloader))
    print('sample_batch:', len(sample_batch), sample_batch[0].size(), sample_batch[1].size(), sample_batch[2].size(),
          sample_batch[0].dtype, sample_batch[3].size(), sample_batch[3].dtype)


    model = MyXLNetModel(args.pretrained_model, num_classes, args.feature_extract, segment_len=args.segment_len)
    model = model.to(device)
    if args.ngpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))

    init_epoch = 0
    # ===================================================================================================== new add
    if args.train_from_scrach is False and len(os.listdir(os.getcwd() + '/' + args.save_dir)) > 0:
        print('*' * 27, 'Loading model weights...')
        ckpt = torch.load(args.save_dir + args.last_new_checkpoint)
        init_epoch = int(args.last_new_checkpoint.split('_')[0][-3:])
        print('*' * 27, 'init_epoch=', init_epoch)
        model_sd = ckpt['net']
        if device.type == 'cuda' and args.ngpu > 1:
            model.module.load_state_dict(model_sd)
        else:
            model.load_state_dict(model_sd)
        print('*' * 27, 'Model loaded success!')
    # =====================================================================================================

    model.eval()
    sample_batch[0] = sample_batch[0].to(device)
    sample_batch[1] = sample_batch[1].to(device)
    sample_batch[2] = sample_batch[2].to(device)

    sample_out = model(sample_batch[0], sample_batch[1], sample_batch[2])
    print('*' * 10, 'sample_out:', sample_out.shape)  # [b, 10]

    print('Params to learn:')
    if args.feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                # print('\t', name)
    else:
        params_to_update = model.parameters()
        # for name, param in model.named_parameters():
        #     if param.requires_grad == True:
        #          print('\t', name)

    #optimizer = torch.optim.Adam(params_to_update, lr=LR, weight_decay=1e-4)
    optimizer = torch.optim.SGD(params_to_update, lr=args.LR, momentum=0.99, weight_decay=1e-4, nesterov=True)

    scheduler_1r = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lr_lambda=lambda epoch: 0.1 if epoch > args.EPOCHS * 0.8 else 1)

    train_model(model, train_dloader, val_dloader, args.save_dir, args.ngpu, optimizer, scheduler_1r,
                init_epoch=init_epoch, num_epochs=args.EPOCHS, print_every=50)

    plot_metric(df_history, 'loss', args.imgs_dir)
    plot_metric(df_history, metric_name, args.imgs_dir)