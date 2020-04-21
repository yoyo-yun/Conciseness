import numpy as np
from collections import Counter, OrderedDict
from torchtext.vocab import Vocab
from torchtext.data import Dataset
import torch
import re

def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def to_categorical_pytorch(label, N=None, device='cpu'):
    if not N:
        N = np.max(label) + 1
    size = list(label.size())
    label = label.view(-1)   # reshape 为向量
    ones = torch.eye(N).to(device)
    ones = ones.index_select(0, label)   # 用上面的办法转为换one hot
    size.append(N)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸
    return ones.view(*size)


def build_text_vocab(*args, **kwargs):
    """Construct the Vocab object for this field from one or more datasets.

    Arguments:
        Positional arguments: Dataset objects or other iterable data
            sources from which to construct the Vocab object that
            represents the set of possible values for this field. If
            a Dataset object is provided, all columns corresponding
            to this field are used; individual columns can also be
            provided directly.
        Remaining keyword arguments: Passed to the constructor of Vocab.
    """
    counter = Counter()
    sources = []
    for arg in args:
        if isinstance(arg, Dataset):
            sources += [getattr(arg, "text")]
        else:
            sources.append(arg)
    for data in sources:
        for x in data:
            # print(x)
            try:
                # counter.update([str(x0) for x0 in x])
                counter.update(x)
            # except TypeError:
            #     counter.update(chain.from_iterable(x))
            except:
                # print(x)
                # print(type(x))
                continue
    specials = list(OrderedDict.fromkeys(
        tok for tok in ["<unk>", "<pad>", None,
                        None] + kwargs.pop('specials', [])
        if tok is not None))
    vocab = Vocab(counter, specials=specials, **kwargs)
    return vocab


def split_sents(string):
    string = re.sub(r"[!?]"," ", string)
    return string.strip().split('.')

def build_nest_vocab(*args, **kwargs):
    """Construct the Vocab object for this field from one or more datasets.

    Arguments:
        Positional arguments: Dataset objects or other iterable data
            sources from which to construct the Vocab object that
            represents the set of possible values for this field. If
            a Dataset object is provided, all columns corresponding
            to this field are used; individual columns can also be
            provided directly.
        Remaining keyword arguments: Passed to the constructor of Vocab.
    """
    counter = Counter()
    sources = []
    for arg in args:
        if isinstance(arg, Dataset):
            sources += [getattr(arg, "text")]
        else:
            sources.append(arg)
    flattened = []
    for source in sources:
        flattened.extend(source)
    for data in flattened:
        for x in data:
            try:
                counter.update(x)
            except:
                continue
    specials = list(OrderedDict.fromkeys(
        tok for tok in ["<unk>", "<pad>", None,
                        None] + kwargs.pop('specials', [])
        if tok is not None))
    vocab = Vocab(counter, specials=specials, **kwargs)
    return vocab