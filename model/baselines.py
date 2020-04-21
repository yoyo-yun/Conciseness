import torch.nn as nn
from .layers.token_emb import TokenEmbedding
from .layers.self_attention import Attention
import torch
import torch.nn.init as init
import torch.nn.functional as F
import math


class LSTM(nn.Module):
    def __init__(self, num_classes, vocab_size, embed_size, pre_embed, lstm_hidden, mask=None, bidirectional=False, is_att=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.pre_embed = pre_embed
        self.feature = None
        self.feature_op = None
        self.mask = mask
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.pre_embed = pre_embed
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.is_att = is_att

        self.embedding = TokenEmbedding(vocab_size, embed_size).cuda()
        self.embedding.weight.data.copy_(self.pre_embed)
        self.embedding.weight.requires_grad = False
        if bidirectional is True:
            print('bidiretional')
            self.lstm = nn.LSTM(self.embed_size, self.lstm_hidden // 2, bidirectional=True, batch_first=True)
        else:
            print('unidirectional')
            self.lstm = nn.LSTM(self.embed_size, self.lstm_hidden, batch_first=True)

        if self.is_att:
            self.att = Attention(self.lstm_hidden)

        self.classifier = nn.Linear(self.lstm_hidden, num_classes)

    def forward(self, seq_token, mask=None):
        seq_embed = self.embedding(seq_token)  # seq_embed(batch, step, dim)
        # print(seq_embed.shape)
        features, (hn, hc) = self.lstm(seq_embed)
        if self.is_att:
            att_patten = self.att(features, mask=mask)
            feature = (features * att_patten.unsqueeze(-1)).sum(1)
        else:
            feature = features[:, -1]
        logits = self.classifier(feature)
        return logits


class StackLSTM(nn.Module):
    def __init__(self, num_classes, vocab_size, embed_size, pre_embed, lstm_hidden, mask=None, bidirectional=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.pre_embed = pre_embed
        self.feature = None
        self.feature_op = None
        self.mask = mask
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.pre_embed = pre_embed
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional

        self.embedding = TokenEmbedding(vocab_size, embed_size)
        self.embedding.weight.data.copy_(self.pre_embed)
        self.embedding.weight.requires_grad = False
        if bidirectional is True:
            print('bidiretional')
            self.lstm1 = nn.LSTM(self.embed_size, self.lstm_hidden // 2, bidirectional=True, batch_first=True)
            self.lstm2 = nn.LSTM(self.embed_size, self.lstm_hidden // 2, bidirectional=True, batch_first=True)
        else:
            print('unidirectional')
            self.lstm1 = nn.LSTM(self.embed_size, self.lstm_hidden, batch_first=True)
            self.lstm2 = nn.LSTM(self.embed_size, self.lstm_hidden, batch_first=True)

        self.classifier = nn.Linear(self.lstm_hidden, num_classes)

    def forward(self, seq_token, mask=None):
        seq_embed = self.embedding(seq_token)  # seq_embed(batch, step, dim)
        features, (hn, hc) = self.lstm1(seq_embed)
        features, (hn, hc) = self.lstm2(features)
        logits = self.classifier(features[:, -1])
        return logits


class CNN(nn.Module):

    def __init__(self,  num_classes, vocab_size, embed_size, output_channel, pre_embed, dropout, **kwargs):
        super().__init__()
        output_channel = output_channel
        target_class = num_classes
        words_num = vocab_size
        words_dim = embed_size
        self.pre_embed = pre_embed
        ks = 3 # There are three conv nets here

        input_channel = 1
        self.embedding = TokenEmbedding(words_num, words_dim).cuda()
        self.embedding.weight.data.copy_(self.pre_embed)
        self.embedding.weight.requires_grad = False

        self.conv1 = nn.Conv2d(input_channel, output_channel, (3, words_dim), padding=(2,0))
        self.conv2 = nn.Conv2d(input_channel, output_channel, (4, words_dim), padding=(3,0))
        self.conv3 = nn.Conv2d(input_channel, output_channel, (5, words_dim), padding=(4,0))

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(ks * output_channel, target_class)

    def forward(self, x, **kwargs):
        x = self.embedding(x).unsqueeze(1) # (batch, channel_input, sent_len, embed_dim)
        x = [F.relu(self.conv1(x)).squeeze(3), F.relu(self.conv2(x)).squeeze(3), F.relu(self.conv3(x)).squeeze(3)]
        # (batch, channel_output, ~=sent_len) * ks
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] # max-over-time pooling
        # (batch, channel_output) * ks
        x = torch.cat(x, 1) # (batch, channel_output * ks)
        x = self.dropout(x)
        logit = self.fc1(x) # (batch, target_size)
        return logit


class CNNLSTM(nn.Module):

    def __init__(self, num_classes, vocab_size, embed_size, output_channel, lstm_hidden, pre_embed, dropout, **kwargs):
        super().__init__()
        output_channel = output_channel
        target_class = num_classes
        words_num = vocab_size
        words_dim = embed_size
        self.pre_embed = pre_embed
        self.lstm_hidden = lstm_hidden
        # ks = 3 # There are three conv nets here

        input_channel = 1
        self.embedding = TokenEmbedding(words_num, words_dim).cuda()
        self.embedding.weight.data.copy_(self.pre_embed)
        self.embedding.weight.requires_grad = False

        self.conv1 = nn.Conv2d(input_channel, output_channel, (3, words_dim), padding=(2,0))
        # self.conv2 = nn.Conv2d(input_channel, output_channel, (4, words_dim), padding=(3,0))
        # self.conv3 = nn.Conv2d(input_channel, output_channel, (5, words_dim), padding=(4,0))
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(output_channel, self.lstm_hidden, batch_first=True)

        self.fc = nn.Linear(self.lstm_hidden, target_class)

    def forward(self, input_sequence, **kwargs):
        x = self.embedding(input_sequence).unsqueeze(1) # (batch, channel_input, sent_len, embed_dim)
        # x = [F.relu(self.conv1(x)).squeeze(3), F.relu(self.conv2(x)).squeeze(3), F.relu(self.conv3(x)).squeeze(3)]
        x = F.relu(self.conv1(x)).squeeze(3)
        # (batch, channel_output, ~=sent_len)
        x = self.dropout(x)
        x, _ = self.lstm(x.contiguous().transpose(1,2))
        logit = self.fc(x[:,-1]) # (batch, target_size)
        return logit


class CapsNet4Sequence(nn.Module):

    def __init__(self, num_classes, vocab_size, embed_size, lstm_hidden, pre_embed, sentence_num,
                 in_dim, out_features, out_dim, **kwargs):
        super().__init__()
        target_class = num_classes
        words_num = vocab_size
        words_dim = embed_size
        self.sentence_num = sentence_num
        self.pre_embed = pre_embed
        self.lstm_hidden = lstm_hidden

        self.embedding = TokenEmbedding(words_num, words_dim)
        self.embedding.weight.data.copy_(self.pre_embed)
        self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(words_dim, self.lstm_hidden // 2, bidirectional=True, batch_first=True)
        self.caps = CapsNet123(self.lstm_hidden, out_features, in_dim, out_dim)
        self.lstm1 = nn.LSTM(out_features * out_dim, self.lstm_hidden // 2, bidirectional=True, batch_first=True)
        self.caps1 = CapsNet123(self.lstm_hidden, out_features, sentence_num, out_dim)

        self.fc = nn.Linear(out_features * out_dim, target_class)

    def forward(self, input_sequence, **kwargs):
        # input_sequence : (batch, sentences, words)
        x = input_sequence.permute(1,0,2) # (sentences, batch, words)

        word_rs = [] # Expect: (sentence, batch, features)

        for i in range(self.sentence_num):
            word_r = self.embedding(x[i,:,:])
            word_r, _ = self.lstm(word_r)
            # x (batch, words, lstm_hidden)
            word_r = word_r.contiguous().permute(0, 2, 1)  # [batch_size, in_features, in_dim]
            word_r = self.caps(word_r)  # [batch, out_features, out_dim]
            word_r = word_r.contiguous().view(x.size(1), -1) # [batch, features]
            word_rs.append(word_r)
            # if word_rs is None:
            #     word_rs = word_r
            # else:
            #     word_rs = torch.cat((word_rs, word_r), 0)
        word_rs = torch.stack(word_rs, 0)
        # print(word_rs.shape)
        word_rs = word_rs.contiguous().permute(1, 0, 2)
        word_rs, _ = self.lstm1(word_rs)
        # x (batch, sent_len, lstm_hidden)
        # print(word_rs.shape)
        word_rs = word_rs.contiguous().permute(0, 2, 1) # [batch_size, in_features, in_dim]
        # print(word_rs.shape)
        word_rs = self.caps(word_rs) # [batch, out_features, out_dim]
        word_rs = word_rs.contiguous().view(word_rs.size(0), -1)
        logit = self.fc(word_rs) # (batch, target_size)
        return logit


class CapsNet(nn.Module):
    """
    input :a group of capsule -> shape:[batch_size*1152(feature_num)*8(in_dim)]
    output:a group of new capsule -> shape[batch_size*10(feature_num)*16(out_dim)]
    """

    def __init__(self, in_features, out_features, in_dim, out_dim):
        """
        """
        super(CapsNet, self).__init__()
        # number of output features,10
        self.out_features = out_features
        # number of input features,1152
        self.in_features = in_features
        # dimension of input capsule
        self.in_dim = in_dim
        # dimension of output capsule
        self.out_dim = out_dim

        # full connect parameter W with shape [1(batch共享),1152,10,8,16]
        self.W = nn.Parameter(torch.randn(1, self.in_features, self.out_features, in_dim, out_dim)).cuda()

    def squash(x, dim):
        # we should do spuash in each capsule.
        # we use dim to select
        sum_sq = torch.sum(x ** 2, dim=dim, keepdim=True)
        sum_sqrt = torch.sqrt(sum_sq)
        return (sum_sq / (1.0 + sum_sq)) * x / sum_sqrt

    def forward(self, x):
        # input x,shape=[batch_size,in_features,in_dim]
        # [batch_size,1152,8]
        # (batch, input_features, in_dim) -> (batch, in_features, out_features,1,in_dim)
        batch_size = x.size(0)
        x = torch.stack([x] * self.out_features, dim=2).unsqueeze(3)

        W = torch.cat([self.W] * batch_size, dim=0)
        # u_hat shape->(batch_size,in_features,out_features,out_dim)=(batch,1152,10,1,16)
        u_hat = torch.matmul(x, W)
        # b for generate weight c,with shape->[1,1152,10,1]
        b = torch.zeros([1, self.in_features, self.out_features, 1]).cuda()
        for i in range(3):
            c = F.softmax(b, dim=2)
            # c shape->[batch_size,1152,10,1,1]
            c = torch.cat([c] * batch_size, dim=0).unsqueeze(dim=4)
            # s shape->[batch_size,1,10,1,16]
            s = (u_hat * c).sum(dim=1, keepdim=True)
            # output shape->[batch_size,1,10,1,16]
            v = self.squash(s, dim=-1)
            v_1 = torch.cat([v] * self.in_features, dim=1)
            # (batch,1152,10,1,16)matmul(batch,1152,10,16,1)->(batch,1152,10,1,1)
            # squeeze
            # mean->(1,1152,10,1)
            # print u_hat.shape,v_1.shape
            update_b = torch.matmul(u_hat, v_1.transpose(3, 4)).squeeze(dim=4).mean(dim=0, keepdim=True)
            b = b + update_b
        return v.squeeze(1).transpose(2, 3)

def squash(x, dim):
    # we should do squash in each capsule.
    # we use dim to select
    sum_sq = torch.sum(x ** 2, dim=dim, keepdim=True)
    sum_sqrt = torch.sqrt(sum_sq)
    return (sum_sq / (1.0 + sum_sq)) * x / sum_sqrt


class CapsNet123(nn.Module):
    """
    input :a group of capsule -> shape:[batch_size*1152(feature_num)*8(in_dim)]
    output:a group of new capsule -> shape[batch_size*10(feature_num)*16(out_dim)]
    """

    def __init__(self, in_features, out_features, in_dim, out_dim):
        """
        """
        super(CapsNet123, self).__init__()
        # number of output features,10
        self.dim_capsule = out_features
        # number of input features,1152
        self.in_features = in_features
        # dimension of input capsule
        self.in_dim = in_dim
        # dimension of output capsule
        self.num_capsule = out_dim
        self.routings = 3

        self.W = nn.Parameter(torch.Tensor(self.dim_capsule*self.num_capsule, self.in_features, 1))
        init.kaiming_uniform(self.W, a=math.sqrt(5))


    def forward(self, u_vecs):
        # u_vecs: (batch, in_features, in_dims)
        batch_size = u_vecs.size(0)
        u_hat_vecs = F.conv1d(u_vecs, self.W) # (batch, ou_features, in_dims)
        u_hat_vecs.permute(0, 2, 1) # (batch, in_dims, out_features)
        input_num_capsule = u_vecs.size(2) # in_dims
        u_hat_vecs = u_hat_vecs.view(batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule)
        u_hat_vecs = u_hat_vecs.permute(0, 2, 1, 3)
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]
        b = torch.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            c = F.softmax(b, dim=1).unsqueeze(-1)   # shape = [None, num_capsule, input_num_capsule, 1]
            s = (u_hat_vecs * c).sum(dim=2, keepdim=True)
            output = squash(s, dim=1) # shape =  [None, num_capsule, 1, dim_capsule]
            if i < self.routings - 1:
                # print(u_hat_vecs.shape)
                # print(output.permute(0,1,3,2).shape)
                b = b + torch.matmul(u_hat_vecs, output.permute(0, 1, 3, 2)).squeeze(3)
        return output.squeeze(2)
