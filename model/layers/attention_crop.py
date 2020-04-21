import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function

kk = 10

print("k index value is " + str(kk))


def generate_mask(length, t_left, t_right):
    return sigmoid_k(length - t_left.view(-1, 1)) - sigmoid_k(length - t_right.view(-1, 1))


def sigmoid_k(inputs, k=kk):
    return 1 / (1 + torch.exp(-inputs * k))


def diff_sigmoid(inputs, k=kk):
    return k * sigmoid_k(inputs) * (1 - sigmoid_k(inputs))


def diff_t(t_left, t_right, length):
    return -diff_sigmoid(length - t_left.view(-1, 1)) + diff_sigmoid(length - t_right.view(-1, 1))


def diff_l(t_left, t_right, length):
    return diff_sigmoid(length - t_left.view(-1, 1)) + diff_sigmoid(length - t_right.view(-1, 1)) - 0.0005


class Crop(Function):
    @staticmethod
    def forward(ctx, t, l, length, mask, training=True):
        np_mask = mask.data.to('cpu').numpy()
        left = torch.from_numpy(np.argmax(np_mask, axis=-1) - 1).float().view(-1, 1).cuda()
        right = torch.from_numpy(length - (np.argmax(np_mask[:, ::-1], axis=-1))).float().view(-1, 1).cuda()
        sentence_length = torch.sum(mask, dim=-1).float().view(-1, 1) / 2.0
        if training:
            ctx.l = l
        l = torch.where(l <= sentence_length, sentence_length, l)
        t_left = (t - l) * ((t - l) >= left).float()
        t_left = torch.where(t_left == 0, left, t_left)
        t_right = ((t + l) <= right).float() * (t + l)
        t_right = torch.where(t_right == 0, right, t_right)
        length_ = torch.arange(0., length, 1.).cuda()
        # ctx.sentence_length = sentence_length
        crop_mask = generate_mask(length_, t_left, t_right)
        crop_mask = (crop_mask >= 0.5).float()
        crop_mask = crop_mask * mask
        if training:
            ctx.sentence_length = sentence_length
            ctx.crop_mask = crop_mask
            ctx.t_left = t_left
            ctx.t_right = t_right
            ctx.length_ = length_
        return crop_mask

    @staticmethod
    def backward(ctx, grad_output):
        attmap = grad_output
        t_left = ctx.t_left
        t_right = ctx.t_right
        length_ = ctx.length_
        crop_mask = ctx.crop_mask
        sentence_length = ctx.sentence_length
        l = ctx.l

        top = torch.max(attmap)
        top = -attmap / top
        # print(attmap[0])
        # print(crop_mask[0])
        # print("ranging is {} - {}".format(t_left[0], t_right[0]))
        # top, _ = torch.max(attmap, dim=-1)
        # top = - attmap / top.view(-1, 1)
        d_t = diff_t(t_left, t_right, length_)
        d_l = diff_l(t_left, t_right, length_)
        d_t = d_t.masked_fill(torch.isnan(d_t), 0)
        d_l = d_l.masked_fill(torch.isnan(d_l), 0)
        d_t = d_t * top
        d_l = d_l * top
        # print(d_t[0])
        # print(d_l[0])
        d_t = d_t.sum(dim=-1).view(-1, 1)
        d_l = d_l.sum(dim=-1).view(-1, 1)
        d_l_zeros = torch.zeros_like(d_l).float().cuda()

        # print(d_t[0])
        # print(d_l[0])
        # print("grad of t is {}; l is {}".format(d_t, d_l))

        d_l = torch.where(l <= sentence_length, d_l_zeros, d_l)

        del crop_mask, ctx.crop_mask, l, sentence_length, d_l_zeros
        torch.cuda.empty_cache()
        return d_t, d_l, None, None, None


crop = Crop.apply


class AttentionMask(nn.Module):
    def __init__(self, in_att):
        super(AttentionMask, self).__init__()
        self.in_att = in_att
        self.learner1 = nn.Linear(self.in_att, 500)
        self.learner2 = nn.Linear(500, 1)
        self.learner3 = nn.Linear(500, 1)

    def forward(self, att, mask=None):
        att_ = torch.tanh(self.learner1(att))
        t, l = torch.relu(self.learner2(att_)), torch.relu(self.learner3(att_))
        return t, l


class AttentionCrop(nn.Module):
    def __init__(self, in_length):
        super(AttentionCrop, self).__init__()
        self.in_length = in_length

    def forward(self, t, l, mask):
        crop_mask = crop(t, l, self.in_length, mask, self.training)
        return crop_mask
