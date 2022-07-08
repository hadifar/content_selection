from torch.utils.data import Dataset
import json
import torch
from itertools import chain

from random import sample, shuffle


def tokenized_data(all_data, data_args, tokenizer):
    padding = "max_length" if data_args.pad_to_max_length else False
    model_inputs = tokenizer([item[0] for item in all_data], max_length=data_args.max_source_length, padding=padding,
                             truncation=True)
    labels = [item[1] for item in all_data]
    return model_inputs, labels


def extract_paragraphs(data, data_args, tokenizer):
    all_data = []
    for c in data:
        positive_labels = []
        for q in c['questions']:
            for aligned_p in q['aligned_paragraphs']:
                if aligned_p.get('annotation'):
                    tmp = [item for item in aligned_p['annotation'].items() if item[1][0] != 'no_support']
                    if len(tmp) > 0:
                        positive_labels.append(aligned_p)

        pos_para = set([p['paragraph_num'] for p in positive_labels])
        neg_para = set(list(range(len(c['tokenized_chapter_text'])))).difference(set(pos_para))

        pos_txt = [(" ".join(" ".join(item) for item in c['tokenized_chapter_text'][pp]), 1) for pp in pos_para]
        neg_txt = [(" ".join(" ".join(item) for item in c['tokenized_chapter_text'][pp]), 0) for pp in neg_para]
        neg_txt = sample(neg_txt, len(pos_txt))
        all_data.append(pos_txt)
        all_data.append(neg_txt)
    all_data = list(chain(*all_data))
    shuffle(all_data)
    inputs, labels = tokenized_data(all_data, data_args, tokenizer)
    return CustomDS(inputs, labels)


def process_data(data, data_args, tknizer, split):
    if data_args.task == 'paragraph_selection':
        return extract_paragraphs(data, data_args, tknizer)
    elif data_args.task == 'question_selection':
        return extract_paragraphs(data, data_args, tknizer)
    else:
        raise Exception('fuck')


def read_json_file(file_path):
    with open(file_path) as outfile:
        data = json.load(outfile)
    return data


def read_data(data_args, tokenizer):
    train_data = read_json_file(data_args.train_file_path)
    valid_data = read_json_file(data_args.valid_file_path)

    if data_args.is_debug_mode == 1:
        train_data = train_data[:1]
        valid_data = valid_data[:1]

    train_ds = process_data(train_data, data_args, tokenizer, 'train')
    valid_ds = process_data(valid_data, data_args, tokenizer, 'valid')

    return train_ds, valid_ds


class CustomDS(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        self.ds_len = len(encodings.encodings)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels is not None:
            item["label"] = torch.tensor([self.labels[idx]],dtype=torch.float)
        return item

    def __len__(self):
        return self.ds_len
