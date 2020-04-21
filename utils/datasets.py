from torchtext.data import Example, Dataset
import pandas as pd


class IMDB(Dataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, fields, **kwargs):
        make_example = Example.fromlist

        pd_reader = pd.read_csv(path, header=None, skiprows=0, encoding="utf-8", sep='\t\t', engine='python')
        usrs = []
        products = []
        labels = []
        texts = []
        for i in range(len(pd_reader[0])):
            usrs.append(pd_reader[0][i])
            products.append(pd_reader[1][i])
            labels.append(pd_reader[2][i])
            texts.append(pd_reader[3][i])

        examples = [make_example([None, None, label, text], fields) for label, text in zip(labels, texts)]

        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        super(IMDB, self).__init__(examples, fields, **kwargs)


class Amazon(Dataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, fields, **kwargs):
        make_example = Example.fromlist

        pd_reader = pd.read_csv(path, header=None, skiprows=1, encoding="utf-8", sep='\t')
        labels = []
        texts = []
        for i in range(len(pd_reader[0])):
        # for i in range(1600):
            try:
                if len(pd_reader[1][i]) is not None:
                    # print(len(pd_reader[1][i]))
                    texts.append(pd_reader[1][i].strip())
                    labels.append(float(pd_reader[2][i]))
            except:
                # print(pd_reader[1][i])
                continue

        examples = [make_example([label, text], fields) for label, text in zip(labels, texts)]

        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        super(Amazon, self).__init__(examples, fields, **kwargs)