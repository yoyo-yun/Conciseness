import torch.nn as nn
import torch
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.features = input_features
        self.linear = nn.Linear(input_features, 1)
        self.p_attn = None

    def forward(self, x, mask=None):
        scores = torch.squeeze(self.linear(x), dim=-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        self.p_attn = p_attn
        return p_attn