import json
from utils import data_loader, losses, eval
from utils.utils_tools import make_print_to_file
from baseline_args import parse_args
import random
from model import baselines
import torch
from torch import nn, optim
import torch.nn.functional as F
import more_itertools

import numpy as np

# args = parse_args()
args = parse_args().parse_args()
params = vars(args)
make_print_to_file(path='./log', file_name="{}_baseline_model_{}".format(args.task, args.id))

print('GPU:', torch.cuda.is_available())
batchsz = args.batch_size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# loading data
# fix_length {imdb:2802; Yelp:1643}
if args.task == "IMDB":
    fix_length = 2802
    args.num_classes = 10
    train_iterator, dev_iterator, test_iterator, pretrained_embedding, pad_idx, example = data_loader.IMDB_torchtxt(
        batchsz=batchsz, device=device, fix_length=None, include_lengths=True, show_config=False)
if args.task == "YELP_13":
    fix_length = 1643
    args.num_classes = 5
    train_iterator, dev_iterator, test_iterator, pretrained_embedding, pad_idx = data_loader.Yelp_13_torchtxt(
            batchsz=batchsz, device=device, fix_length=1643, include_lengths=True, show_config=True)
if args.task == "YELP_14":
    args.num_classes = 5
    fix_length = 1643
    train_iterator, dev_iterator, test_iterator, pretrained_embedding, pad_idx = data_loader.Yelp_14_torchtxt(
        batchsz=batchsz, device=device, fix_length=1643, include_lengths=True, show_config=True)

if args.task == "Amazon":
    args.num_classes = 5
    if args.baseline == "caps":
        fix_length = 2000
        train_iterator, dev_iterator, test_iterator, pretrained_embedding, pad_idx = data_loader.Amazon_hierarchical(
            area=args.area, batchsz=batchsz, device=device, fix_length=fix_length, include_lengths=True,
            show_config=True)
    else:
        train_iterator, dev_iterator, test_iterator, pretrained_embedding, pad_idx = data_loader.Amazon_torchtxt(area=args.area)


# definition of losses and metrics
region_criterion = losses.region_loss
classify_criterion = losses.CrossEntropyLoss
mse_criterion = F.mse_loss
metric_acc = eval.multi_acc
metric_mae = eval.mae
metric_mse = eval.mse

def get_optimizer(models, lr=args.learning_rate):
    optimizer = optim.Adam(
        [
            {'params': filter(lambda p: p.requires_grad, model.parameters())} for model in models
        ],
        lr=lr,
        weight_decay=0)
    return optimizer

if args.baseline == "lstm":
    ## self, num_classes, vocab_size, embed_size, pre_embed, lstm_hidden,  mask = None, bidirectional = False,
    model = baselines.LSTM(args.num_classes, pretrained_embedding.size(0), pretrained_embedding.size(1),
                          pretrained_embedding, args.feature_dim,
                          bidirectional= True if args.num_dir == 1 else False,
                          is_att=args.is_att).to(device)

if args.baseline == "stack":
    ## self, num_classes, vocab_size, embed_size, pre_embed, lstm_hidden,  mask = None, bidirectional = False,
    model = baselines.StackLSTM(args.num_classes, pretrained_embedding.size(0), pretrained_embedding.size(1),
                           pretrained_embedding, args.feature_dim,
                           bidirectional=True if args.num_dir == 1 else False).to(device)

if args.baseline == "cnnlstm":
    # num_classes, vocab_size, embed_size, output_channel, lstm_hidden, pre_embed, dropout,
    model = baselines.CNNLSTM(args.num_classes, pretrained_embedding.size(0), pretrained_embedding.size(1),
                              args.output_channel, args.feature_dim, pretrained_embedding, 0.2)

if args.baseline == "caps":
    # num_classes, vocab_size, embed_size, lstm_hidden, pre_embed, sentence_num, in_dim, out_features, out_dim
    model = baselines.CapsNet4Sequence(args.num_classes, pretrained_embedding.size(0), pretrained_embedding.size(1),
                                       args.feature_dim, pretrained_embedding, 32,
                                       128, args.out_features, args.out_dim)

    # lstm = nn.DataParallel(lstm).to(device)

model = nn.DataParallel(model).to(device)
optimizer = get_optimizer([model])

iteration = more_itertools.ilen(train_iterator)
for epoch in range(args.num_epochs):
    print(">>epoch {}".format(epoch))
    list_acc = []
    list_loss = []
    model.train()
    for i, batch in enumerate(train_iterator):
        if args.baseline == 'caps':
            text, _, _ = batch.text
        else:
            text, _ = batch.text
        label = batch.label
        mask = (text != pad_idx).float()
        logits = model(text, mask=mask)
        loss = classify_criterion(logits, label)
        list_loss.append(loss.item())
        list_acc.append(metric_acc(label, logits).item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # monitoring loss in each epoch
        print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
            i + 1, iteration,
            100 * (i + 1) / iteration,
            loss),
            end="")

        # delete caches
        del text, label, mask, loss
        torch.cuda.empty_cache()

    print('\rtraining phrase: avg_acc is {:.3f} avg_loss is {:.3f}'.
          format(np.array(list_acc).mean(), np.array(list_loss).mean()))

    del list_acc, list_loss
    torch.cuda.empty_cache()
    # exit()
    list_acc = []
    list_mse = []
    list_loss = []
    model.eval()
    for i, batch in enumerate(dev_iterator):
        if args.baseline == 'caps':
            text, _, _ = batch.text
        else:
            text, _ = batch.text
        label = batch.label
        mask = (text != pad_idx).float()
        logits = model(text, mask=mask)
        loss = classify_criterion(logits, label)
        list_loss.append(loss.item())
        list_acc.append(metric_acc(label, logits).item())
        list_mse.append(metric_mse(label, logits).item())

        # delete caches
        del text, label, mask, loss
        torch.cuda.empty_cache()

    print('dev phrase: avg_acc is {:.3f} avg rmse is {:.3f} avg_loss is {:.3f}'.
          format(np.array(list_acc).mean(), np.sqrt(np.array(list_mse).mean()), np.array(list_loss).mean()))

    del list_acc, list_loss
    torch.cuda.empty_cache()

