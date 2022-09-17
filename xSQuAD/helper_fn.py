import json
import random
from itertools import chain
from random import sample, shuffle
import os
import pandas as pd

random.seed(42)


def save_result_on_disk(all_df, result_dic, cache_path, method, task):
    folder_dir = os.path.join(cache_path, method)
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


def cache_exist(cache_path, method, task):
    cache_file = os.path.join(cache_path, method, method + '.{}.csv'.format(task))
    if os.path.isfile(cache_file):
        all_df = [t[1] for t in pd.read_csv(cache_file).groupby('chapter')]
        return all_df
    else:
        return []


def pretty_print_results(cache_path, method, task):
    folder_dir = os.path.join(cache_path, method)
    result_file = os.path.join(folder_dir, method + '.{}.json'.format(task))
    all_results = json.load(open(result_file))
    for item in all_results.items():
        print(item)


def read_json_file(file_path):
    with open(file_path) as outfile:
        data = json.load(outfile)
    return data


def extract_paragraph_pointwise(data, split, negative_samples_weights):
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
            if negative_samples_weights != -1:
                neg_txt = sample(neg_txt, len(pos_txt) * negative_samples_weights)

        all_data.append(pos_txt)
        all_data.append(neg_txt)
    all_data = list(chain(*all_data))
    if split == 'train':
        shuffle(all_data)
    return all_data
