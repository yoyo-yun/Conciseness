import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.data_utils import to_categorical_pytorch

CrossEntropyLoss = nn.CrossEntropyLoss()

m = 0.03
print("margin is " + str(m))


def region_loss(preds2, preds1, y, margin=m):
    preds1 = F.softmax(preds1, dim=-1)
    preds2 = F.softmax(preds2, dim=-1)
    y = to_categorical_pytorch(y, N=preds1.shape[-1], device='cuda')
    pred1 = (preds1 * y).sum(dim=-1)
    pred2 = (preds2 * y).sum(dim=-1)
    return F.margin_ranking_loss(pred2, pred1, torch.ones_like(pred2), margin)


def crossentropylosswithonehot(input, target):
    input = F.log_softmax(input, 1)
    return -(input*target).sum() / len(input)