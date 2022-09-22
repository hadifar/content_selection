import json
from itertools import chain
from random import sample, shuffle

import pandas as pd
import torch
from torch.utils.data import Dataset
import nltk


def tokenized_data(all_data, data_args, tokenizer):
    padding = "max_length" if data_args.pad_to_max_length else False
    model_inputs = tokenizer([item[0] for item in all_data], max_length=data_args.max_source_length, padding=padding,
                             truncation=True)
    labels = [item[1] for item in all_data]
    return model_inputs, labels


def extract_paragraph_from_openstax(data, data_args, split, tokenizer):
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
        if split == 'train':
            neg_txt = _sample_nagatives(pos_txt, neg_txt, factor=2)
        all_data.append(pos_txt)
        all_data.append(neg_txt)
    all_data = list(chain(*all_data))
    if split == 'train':
        shuffle(all_data)
    inputs, labels = tokenized_data(all_data, data_args, tokenizer)
    return CustomDS(inputs, labels), all_data


def _sample_nagatives(pos_sample, neg_sample, factor=2):
    """A naive negative sampling"""
    return sample(neg_sample, k=min(len(pos_sample) * factor, len(neg_sample)))


def extract_question_from_openstax(data, data_args, split, tknizer):
    all_data = []
    for sub_df in data:
        sub_df = sub_df.drop(columns='chapter')
        positive_questions = sub_df[sub_df['label'] == 1].values.tolist()
        negative_questions = sub_df[sub_df['label'] == 0].values.tolist()
        if split == 'train':
            negative_questions = _sample_nagatives(positive_questions, negative_questions, factor=2)

        all_data.append(positive_questions)
        all_data.append(negative_questions)
    all_data = list(chain(*all_data))
    if split == 'train':
        shuffle(all_data)

    inputs, labels = tokenized_data(all_data, data_args, tknizer)
    return CustomDS(inputs, labels), all_data


def extract_paragraph_from_tqa(data, data_args, split, tokenizer):
    all_data = []
    for c in data:
        positive_labels = [s['ground_sentence'] for s in c['questions']]
        all_sentences = list(chain(*[nltk.tokenize.sent_tokenize(p) for p in c['chapter_text_list']]))
        negative_labels = [s for s in all_sentences if s not in positive_labels]

        pos_para = [(pl, 1) for pl in positive_labels]
        neg_para = [(nl, 0) for nl in negative_labels]

        if split == "train":
            neg_para = _sample_nagatives(pos_para, neg_para, factor=2)
            # sample(neg_para, k=min(len(pos_para) * 2, len(neg_para)))

        all_data.append(pos_para)
        all_data.append(neg_para)

    all_data = list(chain(*all_data))
    if split == 'train':
        shuffle(all_data)
    inputs, labels = tokenized_data(all_data, data_args, tokenizer)
    return CustomDS(inputs, labels), all_data


def process_data(data, data_args, tknizer, split):
    if data_args.task == 'openstax_paragraph_selection':
        return extract_paragraph_from_openstax(data, data_args, split, tknizer)
    elif data_args.task == 'openstax_question_selection':
        return extract_question_from_openstax(data, data_args, split, tknizer)
    elif data_args.task == 'tqa_paragraph_selection':
        return extract_paragraph_from_tqa(data, data_args, split, tknizer)
    elif data_args.task == 'tqa_question_selection':
        return
    else:
        raise Exception('fuck')


def read_json_file(file_path):
    with open(file_path) as outfile:
        data = json.load(outfile)
    return data


def read_csv_file(file_path):
    return [g[1] for g in pd.read_csv(file_path).groupby('chapter')]


def subsampling_for_debug(data_args, train_data, valid_data):
    if data_args.is_debug_mode == 1:
        train_data = train_data[:2]
        valid_data = valid_data[:2]
    return train_data, valid_data


def read_data(data_args, tokenizer):
    if data_args.task.find('paragraph_selection') != -1:
        train_data = read_json_file(data_args.train_file_path)
        valid_data = read_json_file(data_args.valid_file_path)

        train_data, valid_data = subsampling_for_debug(data_args, train_data, valid_data)

        train_ds, train_raw_data = process_data(train_data, data_args, tokenizer, 'train')
        valid_ds, valid_raw_data = process_data(valid_data, data_args, tokenizer, 'valid')

        return train_ds, train_raw_data, valid_ds, valid_raw_data
    elif data_args.task.find('question_selection') != -1:
        train_data = read_csv_file(data_args.train_file_path)
        valid_data = read_csv_file(data_args.valid_file_path)

        train_data, valid_data = subsampling_for_debug(data_args, train_data, valid_data)

        train_ds, train_raw_data = process_data(train_data, data_args, tokenizer, 'train')
        valid_ds, valid_raw_data = process_data(valid_data, data_args, tokenizer, 'valid')

        return train_ds, train_raw_data, valid_ds, valid_raw_data
    else:
        raise Exception('task not found ...')


class CustomDS(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        self.ds_len = len(encodings.encodings)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx])
            # item["label"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return self.ds_len
