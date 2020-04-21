import torch
import torch.nn.functional as F


def multi_acc(y, preds):
    """
    get accuracy

    preds: logits
    y: true label
    """
    preds = torch.argmax(F.softmax(preds), dim=1)
    correct = torch.eq(preds, y).float()
    acc = correct.sum() / len(correct)
    return acc


def accuracy(y, preds):
    """
    get accuracy

    preds: normalized logits (by softmax(nD, n>=2) or sigmoid(1D))
    y: true label with one-hot encoder
    """
    preds = torch.argmax(preds, dim=-1)
    correct = torch.eq(preds, torch.argmax(y, dim=1)).float()
    acc = correct.sum() / len(correct)
    return acc


def mae(y, preds):
    preds = F.softmax(preds)
    preds = torch.argmax(preds, 1).squeeze()
    return F.l1_loss(preds.float(), y.float())


def mse(y, preds):
    preds = F.softmax(preds)
    preds = torch.argmax(preds, 1).squeeze()
    return ((preds.float() - y.float()) ** 2).mean()


