from itertools import chain
import random

from torchtext import data
from torchtext.vocab import pretrained_aliases, Vectors

from .datasets import IMDB, Amazon
from .data_utils import *
from torchtext.data.dataset import Dataset
from torchtext.vocab import Vocab

requires = ['batchsz', 'device']


def IMDB_torchtxt(path='./corpus/IMDB', batchsz=32, device='cuda', show_config=True, saving =False, include_lengths=False, fix_length=2802):
    USR = None
    PREDUCT = None
    LABEL = data.LabelField()
    TEXT = data.Field(batch_first=True, tokenize='spacy', include_lengths=include_lengths, fix_length=fix_length)
    fields = [('usr', USR),
              ('product', PREDUCT),
              ('label', LABEL),
              ('text', TEXT),
              ]

    train, dev, test = IMDB.splits(path=path, train='imdb.train.txt.ss', validation='imdb.dev.txt.ss',
                                 test='imdb.test.txt.ss',
                                 fields=fields)

    TEXT.build_vocab(train, vectors='glove.840B.300d')
    LABEL.build_vocab(train)

    train_iterator, dev_iterator, test_iterator = data.BucketIterator.splits(
        (train, dev, test),
        batch_size=batchsz,
        device=device)
    pretrained_embedding = TEXT.vocab.vectors

    examples = []
    examples.append("a tough gunslinger , his brother , and their friends clean up a town full of bad guys . <sssss> good action , though i thought it could have been done a tad more realistically .")
    examples.append("* really movie . <sssss> * not really worth watching . <sssss> * for it , it 's quite boring . <sssss> * there are very few action scence -lrb- just the one that on the t.v.")
    examples.append("yes , you have read correctly . <sssss> the lion king is my number 157 movie . <sssss> it is one of the best of disney ´ s animated films , but not the best . <sssss> it is funny and very well designed . <sssss> i recommend this film to people who like to see a good comedy .")
    examples.append("well , i finally got around to watching `` the mummy '' and i found it quite charming . <sssss> charming insofar as a movie who 's main character is a rotting corpse can be charming . <sssss> what we have here is a big silly old-school adventure with ancient curses and exotic locals . <sssss> it 's lots of fun .")
    examples.append("so now that it 's been about six years , the dust has settled , and the blood has dried , what is there to be said about `` natural born killers ``")
    examples.append("this one was the best , it had the charm , the laughs and the entertainment the whole way through . <sssss> it had a better plot and the little kid is much better than macauly culkin is .")


    example_ = []
    for example in examples:
        example_.append(TEXT.tokenize(example))
    np.save('example.npy', example_)
    example_ = TEXT.process(example_, device='cuda')
    print(example_)
    # exit()

    if show_config:
        print('len of train data:', len(train))  # 67426
        print('len of dev data:', len(dev))  # 8381
        print('len of test data:', len(test))   # 9112

        if include_lengths:
            length = []
            y_list = []
            for batch in train_iterator:
                text, l = batch.text
                y_batch = batch.label
                length = length + l.tolist()
                y_list = y_list + y_batch.tolist()
            for batch in dev_iterator:
                text, l = batch.text
                y_batch = batch.label
                length = length + l.tolist()
                y_list = y_list + y_batch.tolist()
            for batch in test_iterator:
                text, l = batch.text
                y_batch = batch.label
                length = length + l.tolist()
                y_list = y_list + y_batch.tolist()

            # 2802
            print("maximum length is " + str(np.max(np.array(length))))
            # 431.63
            print('average length of datasets is ' + str(np.mean(np.array(length))))

    print('index of "<pad>" is :' + str(TEXT.vocab.stoi.get('<pad>')))
    print('index of "UNK" is :' + str(TEXT.vocab.stoi.get(TEXT.unk_token)))
    pad_idx = TEXT.vocab.stoi.get('<pad>')
    return train_iterator, dev_iterator, test_iterator, pretrained_embedding, pad_idx, example_


def Yelp_13_torchtxt(path='./corpus/Yelp_13', batchsz=32, device='cuda', show_config=True, saving =False, include_lengths=False, fix_length=85):
    USR = None
    PREDUCT = None
    LABEL = data.LabelField()
    TEXT = data.Field(batch_first=True, tokenize='spacy', include_lengths=include_lengths, fix_length=fix_length)
    fields = [('usr', USR),
              ('product', PREDUCT),
              ('label', LABEL),
              ('text', TEXT),
              ]

    train, dev, test = IMDB.splits(path=path, train='yelp-2013-seg-20-20.train.ss', validation='yelp-2013-seg-20-20.dev.ss',
                                 test='yelp-2013-seg-20-20.test.ss',
                                 fields=fields)

    TEXT.build_vocab(train, vectors='glove.840B.300d')
    LABEL.build_vocab(train)

    train_iterator, dev_iterator, test_iterator = data.BucketIterator.splits(
        (train, dev, test),
        batch_size=batchsz,
        device=device)
    pretrained_embedding = TEXT.vocab.vectors

    if show_config:
        print('len of train data:', len(train))  # 62522
        print('len of dev data:', len(dev))  # 7773
        print('len of test data:', len(test))  # 8671

        if include_lengths:
            length = []
            y_list = []
            for batch in train_iterator:
                text, l = batch.text
                y_batch = batch.label
                length = length + l.tolist()
                y_list = y_list + y_batch.tolist()
            for batch in dev_iterator:
                text, l = batch.text
                y_batch = batch.label
                length = length + l.tolist()
                y_list = y_list + y_batch.tolist()
            for batch in test_iterator:
                text, l = batch.text
                y_batch = batch.label
                length = length + l.tolist()
                y_list = y_list + y_batch.tolist()

            # 1643
            print("maximum length is " + str(np.max(np.array(length))))
            # 212.17
            print('average length of datasets is ' + str(np.mean(np.array(length))))

    print('index of "<pad>" is :' + str(TEXT.vocab.stoi.get('<pad>')))
    print('index of "UNK" is :' + str(TEXT.vocab.stoi.get(TEXT.unk_token)))
    pad_idx = TEXT.vocab.stoi.get('<pad>')
    return train_iterator, dev_iterator, test_iterator, pretrained_embedding, pad_idx


def Yelp_14_torchtxt(path='./corpus/Yelp_14', batchsz=32, device='cuda', show_config=True, saving =False, include_lengths=False, fix_length=85):
    USR = None
    PREDUCT = None
    LABEL = data.LabelField()
    TEXT = data.Field(batch_first=True, tokenize='spacy', include_lengths=include_lengths, fix_length=fix_length)
    fields = [('usr', USR),
              ('product', PREDUCT),
              ('label', LABEL),
              ('text', TEXT),
              ]

    train, dev, test = IMDB.splits(path=path, train='yelp-2014-seg-20-20.train.ss', validation='yelp-2014-seg-20-20.dev.ss',
                                 test='yelp-2014-seg-20-20.test.ss',
                                 fields=fields)

    TEXT.build_vocab(train, vectors='glove.840B.300d')
    LABEL.build_vocab(train)

    train_iterator, dev_iterator, test_iterator = data.BucketIterator.splits(
        (train, dev, test),
        batch_size=batchsz,
        device=device)
    pretrained_embedding = TEXT.vocab.vectors

    if show_config:
        print('len of train data:', len(train))  # 183019
        print('len of dev data:', len(dev))  # 22745
        print('len of test data:', len(test))  # 25399

        if include_lengths:
            length = []
            y_list = []
            for batch in train_iterator:
                text, l = batch.text
                y_batch = batch.label
                length = length + l.tolist()
                y_list = y_list + y_batch.tolist()
            for batch in dev_iterator:
                text, l = batch.text
                y_batch = batch.label
                length = length + l.tolist()
                y_list = y_list + y_batch.tolist()
            for batch in test_iterator:
                text, l = batch.text
                y_batch = batch.label
                length = length + l.tolist()
                y_list = y_list + y_batch.tolist()

            # 1643
            print("maximum length is " + str(np.max(np.array(length))))
            # 220.84
            print('average length of datasets is ' + str(np.mean(np.array(length))))

    print('index of "<pad>" is :' + str(TEXT.vocab.stoi.get('<pad>')))
    print('index of "UNK" is :' + str(TEXT.vocab.stoi.get(TEXT.unk_token)))
    pad_idx = TEXT.vocab.stoi.get('<pad>')
    return train_iterator, dev_iterator, test_iterator, pretrained_embedding, pad_idx


def Amazon_torchtxt(path='./corpus/Amazon', area="app", batchsz=32, device='cuda', show_config=True,
                    include_lengths=True, fix_length=None, random_seed = 1234):
    # fix_length = None
    LABEL = data.LabelField(dtype=torch.long)
    TEXT = data.Field(batch_first=True, tokenize='spacy', include_lengths=include_lengths, fix_length=fix_length)
    fields = [
              ('label', LABEL),
              ('text', TEXT),
              ]
    # Apps_for_Android, CDs_and_Vinyl, Electronics, Kindle_Store, Movies_and_TV

    dataset_map = {
        "app": "Apps_for_Android",
        "cds": "CDs_and_Vinyl",
        "kindle": "Kindle_Store",
        "elec": "Electronics",
        "movie": "Movies_and_TV",
        "home":"Home_and_Kitchen",
        "video":"Video_Games",
        "sport":"Sports_and_Outdoors"
    }

    file_path = "reviews_{}_5.tsv".format(dataset_map[area])

    All = Amazon.splits(path=path, train=file_path, fields=fields)[0]

    TEXT.vocab = build_text_vocab(All, vectors='glove.840B.300d')
    # TEXT.build_vocab(All, vectors='glove.840B.300d')
    LABEL.build_vocab(All)

    train, dev, test = All.split(split_ratio=[0.8, 0.1, 0.1], random_state=random.seed(random_seed))

    train_iterator, dev_iterator, test_iterator = data.BucketIterator.splits(
        (train, dev, test),
        batch_size=batchsz,
        device=device)
    pretrained_embedding = TEXT.vocab.vectors

    if show_config:
        print('len of train data:', len(train))
        print('len of dev data:', len(dev))
        print('len of test data:', len(test))

        if include_lengths:
            length = []
            y_list = []
            for batch in train_iterator:
                text, l = batch.text
                y_batch = batch.label
                length = length + l.tolist()
                y_list = y_list + y_batch.tolist()
            for batch in dev_iterator:
                text, l = batch.text
                y_batch = batch.label
                length = length + l.tolist()
                y_list = y_list + y_batch.tolist()
            for batch in test_iterator:
                text, l = batch.text
                y_batch = batch.label
                length = length + l.tolist()
                y_list = y_list + y_batch.tolist()

            # 1643
            print("maximum length is " + str(np.max(np.array(length))))
            # 220.84
            print('average length of datasets is ' + str(np.mean(np.array(length))))

    print('index of "<pad>" is :' + str(TEXT.vocab.stoi.get('<pad>')))
    print('index of "UNK" is :' + str(TEXT.vocab.stoi.get(TEXT.unk_token)))
    pad_idx = TEXT.vocab.stoi.get('<pad>')
    return train_iterator, dev_iterator, test_iterator, pretrained_embedding, pad_idx

def Amazon_hierarchical(path='./corpus/Amazon', area="app", batchsz=32, device='cuda', show_config=True,
                    include_lengths=True, fix_length=None, random_seed = 1234):
    # fix_length = None
    LABEL = data.LabelField(dtype=torch.long)
    NESTING = data.Field(batch_first=True, tokenize='spacy', fix_length=128)
    TEXT = data.NestedField(NESTING, tokenize=split_sents, fix_length=32, include_lengths=include_lengths)
    # NESTING = data.Field(batch_first=True, tokenize='spacy')
    # TEXT = data.NestedField(NESTING, tokenize=split_sents, include_lengths=include_lengths)
    # TEXT = data.Field(batch_first=True, tokenize='spacy', include_lengths=include_lengths, fix_length=fix_length)
    fields = [
              ('label', LABEL),
              ('text', TEXT),
              ]
    # Apps_for_Android, CDs_and_Vinyl, Electronics, Kindle_Store, Movies_and_TV

    dataset_map = {
        "app": "Apps_for_Android",
        "cds": "CDs_and_Vinyl",
        "kindle": "Kindle_Store",
        "elec": "Electronics",
        "movie": "Movies_and_TV",
        "home":"Home_and_Kitchen",
        "video":"Video_Games",
        "sport": "Sports_and_Outdoors"
    }

    file_path = "reviews_{}_5.tsv".format(dataset_map[area])

    All = Amazon.splits(path=path, train=file_path, fields=fields)[0]

    TEXT.vocab = TEXT.nesting_field.vocab = build_nest_vocab(
        All, vectors='glove.840B.300d')
    LABEL.build_vocab(All)

    train, dev, test = All.split(split_ratio=[0.8, 0.1, 0.1], random_state=random.seed(random_seed))

    # print(train.examples[0].text)

    train_iterator, dev_iterator, test_iterator = data.BucketIterator.splits(
        (train, dev, test),
        batch_size=batchsz,
        device=device)
    # pretrained_embedding = TEXT.vocab.vectors
    pretrained_embedding = TEXT.vocab.vectors
    # print(pretrained_embedding)

    if show_config:
        print('len of train data:', len(train))
        print('len of dev data:', len(dev))
        print('len of test data:', len(test))

        if include_lengths:
            length = []
            y_list = []
            for batch in train_iterator:
                text, sentences, words = batch.text
                # print(text[0])
                # print(sentences[0])
                # print(words[0])
                # print(len(batch.text))
                # print(batch.text.size())
                # y_batch = batch.label
                length = length + sentences.tolist()
                # y_list = y_list + y_batch.tolist()
            for batch in dev_iterator:
                text, sentences, words = batch.text
                length = length + sentences.tolist()
            #     text, l = batch.text
            #     y_batch = batch.label
            #     length = length + l.tolist()
            #     y_list = y_list + y_batch.tolist()
            for batch in test_iterator:
                text, sentences, words = batch.text
                length = length + sentences.tolist()
            #     text, l = batch.text
            #     y_batch = batch.label
            #     length = length + l.tolist()
            #     y_list = y_list + y_batch.tolist()

            # 1643
            print("maximum length is " + str(np.max(np.array(length))))
            # 220.84
            print('average length of datasets is ' + str(np.mean(np.array(length))))

    print('index of "<pad>" is :' + str(TEXT.vocab.stoi.get('<pad>')))
    print('index of "UNK" is :' + str(TEXT.vocab.stoi.get(TEXT.unk_token)))
    pad_idx = TEXT.vocab.stoi.get('<pad>')
    return train_iterator, dev_iterator, test_iterator, pretrained_embedding, pad_idx