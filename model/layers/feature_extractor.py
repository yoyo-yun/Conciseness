import torch.nn as nn
from   .token_emb import TokenEmbedding


class FeatureExtractor(nn.Module):
    def __init__(self, vocab_size, embed_size, out_dim, pre_embed=None, drop=None, pad_idx=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.pre_embed = pre_embed
        self.outdim = out_dim
        self.drop = None
        if drop is not None:
            assert 0 < drop < 1
            self.drop = nn.Dropout(p=drop)

        self.embedding = TokenEmbedding(vocab_size, embed_size, pad_idx=pad_idx).cuda()
        self.embedding.weight.data.copy_(self.pre_embed)
        # self.embedding.weight.requires_grad = True
        self.embedding.weight.requires_grad = False

        # self.lstm = nn.LSTM(self.embed_size, self.outdim // 2, bidirectional=True, batch_first=True).cuda()
        self.lstm = nn.LSTM(self.embed_size, self.outdim, batch_first=True).cuda()

    def forward(self, seq_token):
        seq_embed = self.embedding(seq_token) # seq_embed(batch, step, dim)
        # LSTM: (batch, step, in_dim) --> (batch, step, out_dim)
        features, hn = self.lstm(seq_embed)
        if self.drop is not None:
            features = self.drop(features)
        return features, hn




