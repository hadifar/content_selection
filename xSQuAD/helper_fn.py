import json
import random
from itertools import chain
from random import sample, shuffle
import os

import nltk
import pandas as pd

random.seed(42)


def save_result_on_disk(all_df, result_dic, cache_path, dataset_name, method, task):
    folder_dir = os.path.join(cache_path, dataset_name, method)
    print('save file in {}'.format(folder_dir))
    df = pd.concat(all_df)
    if not os.path.isdir(folder_dir):
        os.makedirs(folder_dir)

    cache_file = os.path.join(folder_dir, method + '.{}.csv'.format(task))
    res_file = os.path.join(folder_dir, method + '.{}.json'.format(task))
    # save rankings
    df.to_csv(cache_file, index=False)
    # save results
    json.dump(result_dic, open(res_file, 'w'), indent=4)


def cache_exist(cache_path, dataset_name, method, task):
    cache_file = os.path.join(cache_path, dataset_name, method, method + '.{}.csv'.format(task))
    if os.path.isfile(cache_file):
        all_df = [t[1] for t in pd.read_csv(cache_file).groupby('chapter')]
        return all_df
    else:
        return []


def pretty_print_results(cache_path, dataset_name, method, task):
    folder_dir = os.path.join(cache_path, dataset_name, method)
    result_file = os.path.join(folder_dir, method + '.{}.json'.format(task))
    all_results = json.load(open(result_file))
    for item in all_results.items():
        print(item)


def read_json_file(file_path):
    with open(file_path) as outfile:
        data = json.load(outfile)
    return data


def _sample_nagatives(pos_sample, neg_sample, factor=2):
    """A naive negative sampling"""
    return sample(neg_sample, k=min(len(pos_sample) * factor, len(neg_sample)))


def extract_paragraph_pointwise(train_file_path, dataset_name, split, negative_samples_factor):
    data = read_json_file(train_file_path)
    if dataset_name == 'tqa':
        all_data = []
        for c in data:
            positive_labels = [s['ground_sentence'] for s in c['questions']]
            all_sentences = list(chain(*[nltk.tokenize.sent_tokenize(p) for p in c['chapter_text_list']]))
            negative_labels = [s for s in all_sentences if s not in positive_labels]

            pos_para = [(pl, 1) for pl in positive_labels]
            neg_para = [(nl, 0) for nl in negative_labels]

            if split == "train":
                neg_para = _sample_nagatives(pos_para, neg_para, factor=negative_samples_factor)
                # sample(neg_para, k=min(len(pos_para) * 2, len(neg_para)))

            all_data.append(pos_para)
            all_data.append(neg_para)

        all_data = list(chain(*all_data))
        if split == 'train':
            shuffle(all_data)
        return all_data

    elif dataset_name == 'openstax':
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
                if negative_samples_factor != -1:
                    neg_txt = _sample_nagatives(pos_txt, neg_txt, factor=negative_samples_factor)

            all_data.append(pos_txt)
            all_data.append(neg_txt)
        all_data = list(chain(*all_data))
        if split == 'train':
            shuffle(all_data)
        return all_data
    else:
        raise Exception('dataset not found ...')
