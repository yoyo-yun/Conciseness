import torch.nn as nn
import torch


class Classifier(nn.Module):
    def __init__(self, in_features, out_features, drop_rate=None):
        super().__init__()
        self.feature = in_features
        self.num_classes = out_features
        self.drop_rate = drop_rate
        if drop_rate is not None:
            assert 0 < drop_rate < 1
            self.drop = nn.Dropout(p=drop_rate)
        self.linear = nn.Linear(self.feature, self.num_classes)

    def forward(self, features, p_attns):
        feature = torch.sum(p_attns.contiguous().view(-1, features.shape[-2], 1) * features, -2)
        if self.drop_rate is not None:
            return self.drop(self.linear(feature))
        else:
            return self.linear(feature)


class FusionClassifier(nn.Module):
    def __init__(self, in_features, depth, out_features, drop_rate=None):
        super().__init__()
        self.feature = in_features * depth
        self.num_classes = out_features
        self.drop_rate = drop_rate
        if drop_rate is not None:
            assert 0 < drop_rate < 1
            self.drop = nn.Dropout(p=drop_rate)
        # self.linear_t = nn.Linear(self.feature, in_features)
        # self.linear = nn.Linear(in_features, self.num_classes)
        self.linear = nn.Linear(self.feature, self.num_classes)

    def forward(self, features, p_attns):
        features_ = []
        for feature, p_attn in zip(features, p_attns):
            features_.append(torch.sum(p_attn.contiguous().view(-1, feature.shape[-2], 1) * feature, -2))

        feature_ = torch.cat(features_, dim=-1)
        if self.drop_rate is not None:
            # return self.drop(self.linear(torch.tanh(self.linear_t(feature_))))
            return self.drop(self.linear(feature_))
        else:
            # return self.linear(torch.tanh(self.linear_t(feature_)))
            return self.linear(feature_)




