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

def extract_paragraph_chpaterwise(data, data_args, split, tokenizer):
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

def extract_paragraph_pointwise(data, data_args, split, tokenizer):
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


def extract_paragraph_chapters(data, data_args, split, tokenizer):
    all_data = []
    for c in data:
        positive_labels = []
        for q in c['questions']:
            for aligned_p in q['aligned_paragraphs']:
                if aligned_p.get('annotation'):
                    tmp = [item for item in aligned_p['annotation'].items() if item[1][0] != 'no_support']
                    if len(tmp) > 0:
                        positive_labels.append(aligned_p)

        positive_labels = [pl['paragraph_num'] for pl in positive_labels]

        chapter_text = []
        for ind, ptok in enumerate(c['tokenized_chapter_text']):
            label = int(ind in positive_labels)
            chapter_text.append([" ".join(" ".join(item) for item in ptok), label])
        all_data.append(chapter_text)

    encoded_data = [tokenizer([d[0] for d in item], truncation=True, padding=True, max_length=512) for item in all_data]
    labels = [[t[1] for t in chap] for chap in all_data]

    return CRFDataset(encoded_data, labels)


def process_data(data, data_args, tknizer, split):
    # if data_args.crf == 0:
        # return extract_paragraph_chapters(data, data_args, split, tknizer)
    return extract_paragraph_pointwise(data, data_args, split, tknizer)
    # elif data_args.crf == 1:
    #     return extract_paragraph_chapters(data, data_args, split, tknizer)
    # else:
    #     raise Exception('fuck')


def read_json_file(file_path):
    with open(file_path) as outfile:
        data = json.load(outfile)
    return data


def read_data(data_args, tokenizer):
    train_data = read_json_file(data_args.train_file_path)
    valid_data = read_json_file(data_args.valid_file_path)

    if data_args.is_debug_mode == 1:
        train_data = train_data[:2]
        valid_data = valid_data[:2]

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
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return self.ds_len


class CRFDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        self.ds_len = len(encodings)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v) for k, v in self.encodings[idx].items()}
        if self.labels is not None:
            # item["label"] = torch.tensor([self.labels[idx]])
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return self.ds_len
