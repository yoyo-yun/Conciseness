import torch.nn as nn


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=300, pad_idx=1):
        super().__init__(vocab_size, embed_size, padding_idx=pad_idx)