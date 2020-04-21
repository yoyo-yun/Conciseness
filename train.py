import json
import random
from utils import data_loader, losses, eval
from utils.utils_tools import make_print_to_file
import torch
from model.layers import feature_extractor, self_attention, classifier, attention_crop
from torch import nn, optim
import torch.nn.functional as F
import more_itertools
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='IMDB',
                        help='Options: IMDB, YELP_13, YELP_14, Amazon')
    parser.add_argument('--area', type=str, default='app',
                        help='Options: app, elec, kindle, cds, movie')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--feature_dim', type=int, default=250)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--id', type=str, default='test')
    parser.add_argument('--depth', type=int, default=3),
    parser.add_argument('--using_extend', type=int, default=0, choices=[0, 1]),
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--layer', type=int, default=0, choices=[0, 1, 2],
                        help='choices of which layer would you want to work')
    parser.add_argument('--step', type=str,
                        choices=['base', 'att', 'scale', 'att2', 'fusion', 'eval', 'single'],
                        help='choices of steps in training process')
    return parser.parse_args()


args = parse_args()
params = vars(args)
make_print_to_file(path='./log', file_name="{}_RA-LSTM_model_{}".format(args.area if args.task == "Amazon" else args.task, args.id))

print('GPU:', torch.cuda.is_available())
batchsz = args.batch_size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# torch.manual_seed(args.seed)
# np.random.seed(args.seed)
# random.seed(args.seed)

# loading data
# fix_length {imdb:2802; Yelp:1643}
if args.task == "IMDB":
    fix_length = 2802
    args.num_classes = 10
    train_iterator, dev_iterator, test_iterator, pretrained_embedding, pad_idx, example = data_loader.IMDB_torchtxt(
        batchsz=batchsz, device=device, fix_length=2802, include_lengths=True, show_config=False)
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
    # fix_length 2000
    fix_length = 2000
    train_iterator, dev_iterator, test_iterator, pretrained_embedding, pad_idx = data_loader.Amazon_torchtxt(
        area=args.area, batchsz=batchsz, device=device, fix_length=2000, include_lengths=True, show_config=True, random_seed=args.seed)


# echo hyper_parameters
print(json.dumps(params, indent=2))
print("vocabulary size is "+ str(pretrained_embedding.size(0)))


# definition of losses and metrics
region_criterion = losses.region_loss
classify_criterion = losses.CrossEntropyLoss
mse_criterion = F.mse_loss
metric_acc = eval.multi_acc
metric_mae = eval.mae
metric_mse = eval.mse


class CheckPoint(object):
    def __init__(self, saved_path='./saved_model/test.pth', layer=0):
        super().__init__()
        self.best_score = 0
        self.history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}
        self.cur_loss = 0
        self.cur_acc = 0
        self.cur_val_loss = 0
        self.cur_val_acc = 0
        self.layer = layer
        self.saved_path = saved_path

    def checkpoint(self, path=None):
        if self.cur_val_acc > self.best_score:
            self.best_score = self.cur_val_acc
            if path is None:
                path = self.saved_path
            if self.layer == 0:
                save_base_model(path)
            else:
                save_models(path)

    def loadpoint(self, path=None):
        if path is None:
            path = self.saved_path
        load_models(path)

    def show(self):
        for (k, v) in self.history.items():
            print("dict[%s]=" % k, v)

    def update_loss(self, loss):
        self.cur_loss = loss
        self.history['loss'].append(loss)

    def update_acc(self, acc):
        self.cur_acc = acc
        self.history['acc'].append(acc)

    def update_val_loss(self, val_loss):
        self.cur_val_loss = val_loss
        self.history['val_loss'].append(val_loss)

    def update_val_acc(self, val_acc):
        self.cur_val_acc = val_acc
        self.history['val_acc'].append(val_acc)


def get_attention_layer():
    att = self_attention.Attention(input_features=args.feature_dim)
    att = att.to(device)
    # att = nn.DataParallel(att).to(device)
    return att


def get_base_model_layer():
    f_ext_ = feature_extractor.FeatureExtractor(vocab_size=pretrained_embedding.shape[0],
                                                embed_size=pretrained_embedding.shape[1],
                                                out_dim=args.feature_dim,
                                                pre_embed=pretrained_embedding,
                                                drop=None,
                                                pad_idx=pad_idx).to(device)
    return f_ext_


def get_crop_layer():
    att_crop = attention_crop.AttentionMask(fix_length).to(device)
    crop_m = attention_crop.AttentionCrop(fix_length).to(device)
    # att_crop = nn.DataParallel(att_crop).to(device)
    return att_crop, crop_m


def get_classifier_layer():
    clc = classifier.Classifier(in_features=args.feature_dim, out_features=args.num_classes)
    # clc = nn.DataParallel(clc).to(device)
    clc = clc.to(device)
    return clc


def get_fusion_classifier_layer():
    fusion = classifier.FusionClassifier(in_features=args.feature_dim, depth=args.depth, out_features=args.num_classes)
    # fusion = nn.DataParallel(fusion).to(device)
    fusion = fusion.to(device)
    return fusion


def get_optimizer(models, lr=args.learning_rate):
    optimizer = optim.Adam(
        [
            {'params': filter(lambda p: p.requires_grad, model.parameters())} for model in models
        ],
        lr=lr,
        weight_decay=0)
    return optimizer


def init_base_model(f_exts, att_exts, clcs, depth, data_iterator, optimizer):
    iteration = more_itertools.ilen(data_iterator)
    for epoch in range(args.num_epochs):
        print(">>epoch {}".format(epoch))
        list_acc = []
        list_loss = []
        for _ in range(depth):
            list_acc.append([])
            list_loss.append([])
        for model in f_exts:
            model.train()
        for model in att_exts:
            model.train()
        for model in clcs:
            model.train()
        for i, batch in enumerate(data_iterator):
            loss = torch.tensor(0.).cuda()
            text, _ = batch.text
            label = batch.label
            mask = (text != pad_idx).float()
            feature, _ = f_ext(text)
            for j in range(depth):
                p_attn = att_exts[j](feature, mask=mask)
                pred = clcs[j](feature, p_attn)
                loss = loss + classify_criterion(pred, label)
                list_loss[j].append(classify_criterion(pred, label).item())
                list_acc[j].append(metric_acc(label, pred).item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # monitoring loss in each epoch
            print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                i + 1, iteration,
                100 * (i + 1) / iteration,
                loss),
                end="")
        for j in range(depth):
            print('\rtraining phrase at scale {}: avg_acc is {:.3f}; avg_loss is {:.3f}'.
                  format(j, np.array(list_acc[j]).mean(), np.array(list_loss[j]).mean()))


def init_crop_layer(f_exts, att_exts, clcs, att_crops, crop_ms, layer, data_iterator, optimizer):
    iteration = more_itertools.ilen(data_iterator)
    for model in f_exts:
        model.train()
    for model in att_exts:
        model.train()
    for model in clcs:
        model.train()
    for model in att_crops:
        model.train()
    for model in crop_ms:
        model.eval()
    for epoch in range(3):
        print(">>epoch {}".format(epoch))
        avg_loss = []
        for i, batch in enumerate(data_iterator):
            text, _ = batch.text
            mask = (text != pad_idx).float()
            t, l = None, None
            p_attn = None

            for j in range(layer):
                feature, _ = f_exts[j](text)
                p_attn = att_exts[j](feature, mask=mask.detach())
                t, l = att_crops[j](p_attn, mask)
                mask = crop_ms[j](t, l, mask)

            label_t = torch.argmax(p_attn, dim=-1).float().view(-1, 1).cuda()
            label_l = torch.tensor([0.6 / (layer + 1) * fix_length]).repeat(t.shape[0], 1).cuda()
            loss = mse_criterion(t, label_t) + mse_criterion(l, label_l)
            avg_loss.append(float(loss.item()))

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
            del text, mask, p_attn, t, l, loss, label_t, label_l
            torch.cuda.empty_cache()

        avg_loss = np.array(avg_loss).mean()
        print('\rinitialing ATTN_{} average loss is : {:.3f}'.format(layer, avg_loss))


def refine_scale(f_exts, att_exts, clcs, att_crops, crop_ms, layer, data_iterator, optimizer, att_optimizer, callbacks=None):
    iteration = more_itertools.ilen(data_iterator)
    for epoch in range(args.num_epochs):
        print(">>epoch {}".format(epoch))
        list_acc = []
        list_loss = []
        length = []
        for _ in range(layer+1):
            list_acc.append([])
            list_loss.append([])
            length.append([])
        for model in f_exts:
            model.train()
        for model in att_exts:
            model.train()
        for model in clcs:
            model.train()
        for model in att_crops:
            model.train()
        for model in crop_ms:
            model.train()
        for i, batch in enumerate(data_iterator):
            p_attns = []
            masks = []
            preds = []
            loss = torch.tensor(0.).cuda()
            text, _ = batch.text
            label = batch.label
            mask = (text != pad_idx).float()
            masks.append(mask)

            for j in range(layer):
                feature, _ = f_exts[j](text)
                p_attn = att_exts[j](feature, mask=mask.detach())
                p_attns.append(p_attn)
                pred = clcs[j](feature, p_attn)
                preds.append(pred)
                t, l = att_crops[j](p_attn, mask)
                mask = crop_ms[j](t, l, mask)
                masks.append(mask)

            feature, _ = f_exts[layer](text)
            p_attn = att_exts[layer](feature, mask=mask.detach())
            p_attns.append(p_attn)
            pred = clcs[layer](feature, p_attn)
            preds.append(pred)

            # calculating loss and metrics
            for j in range(layer+1):
                # loss = classify_criterion(preds[j], label)
                list_loss[j].append(classify_criterion(preds[j], label).item())
                list_acc[j].append(metric_acc(label, preds[j]).item())
                length[j].append((masks[j].sum() / len(mask)).item())
            for j in range(layer):
                loss = loss + region_criterion(preds[j+1], preds[j], label)

            # updating parameters
            att_optimizer.zero_grad()
            masks[layer].backward(p_attns[layer], retain_graph=True)
            att_optimizer.step()

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
            del text, label, mask, feature, p_attns, masks, preds, loss
            torch.cuda.empty_cache()

        for j in range(layer+1):
            print('\rtraining phrase at scale {}: avg_acc is {:.3f} avg_loss is {:.3f} avg_length is {:.3f}'.
                  format(j, np.array(list_acc[j]).mean(), np.array(list_loss[j]).mean(), np.array(length[j]).mean()))

        if callbacks is not None:
            callbacks.update_acc(np.array(list_acc[layer]).mean())
            callbacks.update_loss(np.array(list_loss[layer]).mean())
        del list_acc, list_loss, length
        torch.cuda.empty_cache()

        # # evaluation
        list_acc = []
        list_loss = []
        for _ in range(layer + 1):
            list_acc.append([])
            list_loss.append([])
        for model in f_exts:
            model.train()
        for model in att_exts:
            model.eval()
        for model in clcs:
            model.eval()
        for model in att_crops:
            model.eval()
        for model in crop_ms:
            model.eval()
        for i, batch in enumerate(dev_iterator):
            p_attns = []
            masks = []
            preds = []
            text, _ = batch.text
            label = batch.label
            mask = (text != pad_idx).float()

            for j in range(layer):
                feature, _ = f_exts[j](text)
                p_attn = att_exts[j](feature, mask=mask.detach())
                p_attns.append(p_attn)
                pred = clcs[j](feature, p_attn)
                preds.append(pred)
                t, l = att_crops[j](p_attn, mask)
                mask = crop_ms[j](t, l, mask)
                masks.append(mask)

            feature, _ = f_exts[layer](text)
            p_attn = att_exts[layer](feature, mask=mask.detach())
            p_attns.append(p_attn)
            pred = clcs[layer](feature, p_attn)
            preds.append(pred)

            # calculating loss and metrics
            for j in range(layer + 1):
                list_acc[j].append(metric_acc(label, preds[j]).item())

            # delete caches
            del text, label, mask, feature, p_attns, preds, masks
            torch.cuda.empty_cache()

        # monitoring metrics after each epoch
        for j in range(layer+1):
            print('\rdev phrase acc at scale {}: {:.3f}'.format(j, np.array(list_acc[j]).mean()))

        if callbacks is not None:
            callbacks.update_val_acc(np.array(list_acc[layer]).mean())
            callbacks.checkpoint()
        del list_acc
        torch.cuda.empty_cache()


def init_fusion(data_iterator, optimizer, callbacks=None):
    iteration = more_itertools.ilen(data_iterator)
    f_ext.eval()
    f_ext0.eval()
    f_ext1.eval()
    f_ext2.eval()
    att_ext0.eval()
    att_ext1.eval()
    att_ext2.eval()
    clc0.eval()
    clc1.eval()
    clc2.eval()
    att_crop1.eval()
    crop_m1.eval()
    att_crop2.eval()
    crop_m2.eval()
    for epoch in range(args.num_epochs):
        print(">>epoch {}".format(epoch))
        list_acc = []
        list_loss = []
        fusion_clc.train()
        for i, batch in enumerate(data_iterator):
            text, _ = batch.text
            label = batch.label
            mask = (text != pad_idx).float()
            feature0, _ = f_ext0(text)
            feature1, _ = f_ext1(text)
            feature2, _ = f_ext2(text)
            p_attn0 = att_ext0(feature0, mask=mask)
            t1, l1 = att_crop1(p_attn0, mask)
            crop_mask1 = crop_m1(t1, l1, mask)
            p_attn1 = att_ext1(feature1, mask=crop_mask1)
            t2, l2 = att_crop1(p_attn1, crop_mask1)
            crop_mask2 = crop_m1(t2, l2, crop_mask1)
            p_attn2 = att_ext2(feature2, mask=crop_mask2)
            if args.depth == 3:
                pred = fusion_clc([feature0.detach(), feature1.detach(), feature2.detach()],
                                  [p_attn0.detach(), p_attn1.detach(), p_attn2.detach()])
            if args.depth == 2:
                pred = fusion_clc([feature0.detach(), feature1.detach()],
                                  [p_attn0.detach(), p_attn1.detach()])

            # calculating loss and metrics
            loss = classify_criterion(pred, label)
            list_loss.append(loss.item())
            list_acc.append(metric_acc(label, pred).item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # monitoring loss in each epoch
            print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                i + 1, iteration,
                100 * (i + 1) / iteration,
                loss),
                end="")

            del text, label, mask, feature0, feature1, feature2, p_attn0, p_attn1, p_attn2, pred
            torch.cuda.empty_cache()

            # monitoring metrics after each epoch
        print('\rtraining phrase: avg_acc is {:.3f}; avg_loss is {:.3f}'.
              format(np.array(list_acc).mean(), np.array(list_loss).mean()))

        if callbacks is not None:
            callbacks.update_acc(np.array(list_acc).mean())
            callbacks.update_loss(np.array(list_loss).mean())
        del list_acc, list_loss
        torch.cuda.empty_cache()

        # # evaluation
        list_acc = []
        list_mae = []
        list_rmse = []
        fusion_clc.train()
        for i, batch in enumerate(test_iterator):
            text, _ = batch.text
            label = batch.label
            mask = (text != pad_idx).float()
            feature0, _ = f_ext0(text)
            feature1, _ = f_ext1(text)
            feature2, _ = f_ext2(text)
            p_attn0 = att_ext0(feature0, mask=mask)
            t1, l1 = att_crop1(p_attn0, mask)
            crop_mask1 = crop_m1(t1, l1, mask)
            p_attn1 = att_ext1(feature1, mask=crop_mask1)
            t2, l2 = att_crop1(p_attn1, crop_mask1)
            crop_mask2 = crop_m1(t2, l2, crop_mask1)
            p_attn2 = att_ext2(feature2, mask=crop_mask2)
            if args.depth == 3:
                pred = fusion_clc([feature0.detach(), feature1.detach(), feature2.detach()],
                                  [p_attn0.detach(), p_attn1.detach(), p_attn2.detach()])
            if args.depth == 2:
                pred = fusion_clc([feature0.detach(), feature1.detach()],
                                  [p_attn0.detach(), p_attn1.detach()])

            # calculating metrics
            list_acc.append(metric_acc(label, pred).item())
            list_mae.append(metric_mae(label, pred).item())
            list_rmse.append(metric_mse(label, pred).item())

            del text, label, mask, feature0, feature1, feature2, p_attn0, p_attn1, p_attn2, pred
            torch.cuda.empty_cache()

        # monitoring metrics after each epoch
        print('\rdev phrase acc: {:.3f} mae : {:.3f} rmse: {:.3f}'.
              format(np.array(list_acc).mean(), np.array(list_mae).mean(), np.sqrt(np.array(list_rmse).mean())))

        if callbacks is not None:
            callbacks.update_val_acc(np.array(list_acc).mean())
            callbacks.checkpoint()
        del list_acc
        torch.cuda.empty_cache()


def train_base_model(saving=True):
    print("training based model ...".center(60, "-"))
    temp_optimizer = get_optimizer([f_ext, att_ext0, clc0, att_ext1, clc1, att_ext2, clc2])
    init_base_model([f_ext],
                    [att_ext0, att_ext1, att_ext2],
                    [clc0, clc1, clc2],
                    args.depth, train_iterator, temp_optimizer)
    if saving:
        file_name = './saved_model/{}_base_model_{}.pth'.format(args.task, args.id)
        save_base_model(file_name)
    print("training based model done !".center(60, "-"))


def train_scale(layer, using_extend=False):
    """
    :param layer: which layer work on
    :param using_extend: whether to load a external model to init the model
    :return: None
    """
    print("refine scale {} ...".format(layer).center(60, "-"))
    file_names = [
        './saved_model/{}_base_model_{}.pth'.format(args.task, args.id),
        './saved_model/{}_base_model_scale_1_{}.pth'.format(args.task, args.id),
        './saved_model/{}_base_model_scale_2_{}.pth'.format(args.task, args.id)
    ]
    load_models(file_names[layer])

    if using_extend:
        file_name_att = './saved_model/{}_single_att_model_{}.pth'.format(args.task, args.id)
        load_model = torch.load(file_name_att)
        if layer == 1:
            f_ext1.load_state_dict(load_model['f_ext0'])
            att_ext1.load_state_dict(load_model['att_ext0'])
            clc1.load_state_dict(load_model['clc0'])
        if layer == 2:
            f_ext2.load_state_dict(load_model['f_ext0'])
            att_ext2.load_state_dict(load_model['att_ext0'])
            clc2.load_state_dict(load_model['clc0'])

    checkpoint = CheckPoint(saved_path=file_names[layer], layer=layer)

    temp_optimizers = [
        get_optimizer([f_ext1, att_ext1, clc1]),
        get_optimizer([f_ext2, att_ext2, clc2])
    ]

    temp_att_optimizers = [
        get_optimizer([att_crop1]),
        get_optimizer([att_crop2])
    ]

    refine_scale([f_ext0, f_ext1, f_ext2],
                 [att_ext0, att_ext1, att_ext2],
                 [clc0, clc1, clc2],
                 [att_crop1, att_crop2],
                 [crop_m1, crop_m2],
                 layer, train_iterator, temp_optimizers[layer-1], temp_att_optimizers[layer-1], callbacks=checkpoint)

    print("refine scale {} ...".format(layer).center(60, "-"))


def train_att(layer, saving=False):
    print("initialing Att-{} ...".format(layer).center(60, "-"))
    file_names = [
        './saved_model/{}_base_model_{}.pth'.format(args.task, args.id),
        './saved_model/{}_base_model_scale_1_{}.pth'.format(args.task, args.id),
        './saved_model/{}_base_model_scale_2_{}.pth'.format(args.task, args.id)
    ]
    load_models(file_names[layer-1])

    temp_att_optimizers = [
        get_optimizer([att_crop1], lr=1e-2),
        get_optimizer([att_crop2], lr=1e-2)
    ]

    init_crop_layer([f_ext0, f_ext1, f_ext2],
                    [att_ext0, att_ext1, att_ext2],
                    [clc0, clc1, clc2],
                    [att_crop1, att_crop2],
                    [crop_m1, crop_m2],
                    layer, train_iterator, temp_att_optimizers[layer-1])

    if saving:
        file_name = './saved_model/{}_base_model_scale_{}_{}.pth'.format(args.task, layer, args.id)
        save_models(file_name)

    print("initialing Att-{} done !".format(layer).center(60, "-"))


def train_single_att(saving=True):
    temp_optimizer = get_optimizer([f_ext, att_ext0, clc0])
    args.depth = 1
    init_base_model([f_ext],
                    [att_ext0],
                    [clc0],
                    args.depth, train_iterator, temp_optimizer)
    if saving:
        file_name = './saved_model/{}_single_att_model_{}.pth'.format(args.task, args.id)
        save_base_model(file_name)


def train_fusion():
    print("training fusion classification ...".center(60, "-"))
    file_name = './saved_model/{}_base_model_scale_2_{}.pth'.format(args.task, args.id)
    load_models(file_name)
    checkpoint = CheckPoint(saved_path='./saved_model/{}_fusion_{}.pth', layer=2)
    temp_optimizer = get_optimizer([fusion_clc])
    init_fusion(train_iterator, temp_optimizer, callbacks=checkpoint)
    print("training fusion classification done !".center(60, "-"))


def save_models(file_name):
    state = {'f_ext': f_ext.state_dict(),
             'f_ext0': f_ext0.state_dict(),
             'f_ext1': f_ext1.state_dict(),
             'f_ext2': f_ext2.state_dict(),
             'att_ext0': att_ext0.state_dict(),
             'clc0': clc0.state_dict(),
             'att_ext1': att_ext1.state_dict(),
             'clc1': clc1.state_dict(),
             'att_ext2': att_ext2.state_dict(),
             'clc2': clc2.state_dict(),
             'att_crop1': att_crop1.state_dict(),
             'crop_m1': crop_m1.state_dict(),
             'att_crop2': att_crop2.state_dict(),
             'crop_m2': crop_m2.state_dict(),
             'fusion': fusion_clc.state_dict()
             }
    torch.save(state, file_name)


def save_base_model(file_name):
    state = {'f_ext': f_ext.state_dict(),
             'f_ext0': f_ext.state_dict(),
             'f_ext1': f_ext.state_dict(),
             'f_ext2': f_ext.state_dict(),
             'att_ext0': att_ext0.state_dict(),
             'clc0': clc0.state_dict(),
             'att_ext1': att_ext1.state_dict(),
             'clc1': clc1.state_dict(),
             'att_ext2': att_ext2.state_dict(),
             'clc2': clc2.state_dict(),
             'att_crop1': att_crop1.state_dict(),
             'crop_m1': crop_m1.state_dict(),
             'att_crop2': att_crop2.state_dict(),
             'crop_m2': crop_m2.state_dict(),
             'fusion': fusion_clc.state_dict()
             }
    torch.save(state, file_name)


def load_models(file_name):
    load_model = torch.load(file_name)
    f_ext.load_state_dict(load_model['f_ext'])
    f_ext0.load_state_dict(load_model['f_ext0'])
    f_ext1.load_state_dict(load_model['f_ext1'])
    f_ext2.load_state_dict(load_model['f_ext2'])
    att_ext0.load_state_dict(load_model['att_ext0'])
    clc0.load_state_dict(load_model['clc0'])
    att_ext1.load_state_dict(load_model['att_ext1'])
    clc1.load_state_dict(load_model['clc1'])
    att_ext2.load_state_dict(load_model['att_ext2'])
    clc2.load_state_dict(load_model['clc2'])
    att_crop1.load_state_dict(load_model['att_crop1'])
    crop_m1.load_state_dict(load_model['crop_m1'])
    att_crop2.load_state_dict(load_model['att_crop2'])
    crop_m2.load_state_dict(load_model['crop_m2'])
    fusion_clc.load_state_dict(load_model['fusion'])


def evaluation(statistic=False):
    print(" evaluation start ... ".center(60, "-"))
    file_name = "./saved_model/{}_fusion_{}.pth"
    load_models(file_name)
    f_ext.eval()
    f_ext0.eval()
    f_ext1.eval()
    f_ext2.eval()
    att_ext0.eval()
    att_ext1.eval()
    att_ext2.eval()
    clc0.eval()
    clc1.eval()
    clc2.eval()
    att_crop1.eval()
    crop_m1.eval()
    att_crop2.eval()
    crop_m2.eval()

    # evaluation
    acc = []
    length = []
    preds_list = []
    list_acc = []
    list_rmse = []
    list_mae = []
    for __ in range(args.depth):
        acc.append([])
        length.append([])
        preds_list.append([])

    for i, batch in enumerate(test_iterator):
        text, _ = batch.text
        label = batch.label
        mask = (text != pad_idx).float()

        feature0, _ = f_ext0(text)
        feature1, _ = f_ext1(text)
        feature2, _ = f_ext2(text)
        p_attn0 = att_ext0(feature0, mask=mask)
        pred0 = clc0(feature0, p_attn0)
        t1, l1 = att_crop1(p_attn0, mask)
        crop_mask1 = crop_m1(t1, l1, mask)
        p_attn1 = att_ext1(feature1, mask=crop_mask1)
        pred1 = clc1(feature1, p_attn1)
        t2, l2 = att_crop1(p_attn1, crop_mask1)
        crop_mask2 = crop_m1(t2, l2, crop_mask1)
        p_attn2 = att_ext2(feature2, mask=crop_mask2)
        pred2 = clc2(feature2, p_attn2)
        if args.depth == 3:
            pred = fusion_clc([feature0.detach(), feature1.detach(), feature2.detach()],
                              [p_attn0.detach(), p_attn1.detach(), p_attn2.detach()])
        if args.depth == 2:
            pred = fusion_clc([feature0.detach(), feature1.detach()],
                              [p_attn0.detach(), p_attn1.detach()])

        # calculating metrics
        acc[0].append(metric_acc(label, pred0).item())
        acc[1].append(metric_acc(label, pred1).item())
        acc[2].append(metric_acc(label, pred2).item())
        list_acc.append(metric_acc(label, pred).item())
        list_mae.append(metric_mae(label, pred).item())
        list_rmse.append(metric_mse(label, pred).item())

        # static the length of every scale
        if statistic:
            length[0].append((mask.sum() / len(mask)).item())
            length[1].append((crop_mask1.sum() / len(crop_mask1)).item())
            length[2].append((crop_mask2.sum() / len(crop_mask2)).item())

        del text, label, mask, feature0, feature1, feature2, p_attn0, p_attn1, p_attn2, pred0, pred1, pred2, pred
        torch.cuda.empty_cache()

    # monitoring metrics after each epoch
    print('\rtest phrase acc : {:.3f}'.format(np.array(list_rmse).mean().sqrt()))
    if statistic:
        for j in range(args.depth):
            print('\rdev phrase average length at scale {}: {:.3f}'.format(j, np.array(length[j]).mean()))

    del acc
    torch.cuda.empty_cache()

    print(" evaluation end ! ".center(60, "-"))


if __name__ == '__main__':
    f_ext = get_base_model_layer()
    f_ext0 = get_base_model_layer()
    f_ext1 = get_base_model_layer()
    f_ext2 = get_base_model_layer()
    att_ext0 = get_attention_layer()
    clc0 = get_classifier_layer()
    att_ext1 = get_attention_layer()
    clc1 = get_classifier_layer()
    att_ext2 = get_attention_layer()
    clc2 = get_classifier_layer()
    att_crop1, crop_m1 = get_crop_layer()
    att_crop2, crop_m2 = get_crop_layer()
    fusion_clc = get_fusion_classifier_layer()

    if args.step == "base":
        train_base_model()
    if args.step == 'att':
        train_att(layer=args.layer, saving=True)
    if args.step == 'scale':
        if args.using_extend == 0:
            args.using_extend = False
        else:
            args.using_extend = True
        train_scale(layer=args.layer, using_extend=args.using_extend)
    if args.step == 'eval':
        evaluation(statistic=True)
    if args.step == 'fusion':
        train_fusion()
    if args.step == 'single':
        train_single_att(saving=True)

    print(" having a good look! ".center(60, '*'))
