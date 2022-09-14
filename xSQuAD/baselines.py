import argparse
import json
import random

import nltk.tokenize
import pandas as pd
from lexrank import LexRank, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from textstat import textstat

from xSQuAD.utils import calculate_all_score, read_json_file, extract_paragraph_pointwise


def load_lexrank(train_file_path):
    with open(train_file_path) as inpfile:
        raw_dataset = json.load(inpfile)
        lexrank = LexRank([item['intro'] + ' ' + item['chapter_text'] for item in raw_dataset],
                          stopwords=STOPWORDS['en'])
        return lexrank


def load_svm(train_file_path):
    train_data = extract_paragraph_pointwise(read_json_file(train_file_path), 'train')
    # valid_data = extract_paragraph_pointwise(read_json_file('raw_data/qg_valid.json')[:2], 'valid')
    x_train, y_train = [str(item[0]) for item in train_data], [item[1] for item in train_data]
    # x_valid, y_valid = [str(item[0]) for item in train_data], [item[1] for item in train_data]
    pipe = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 2))), ('svc', SVC(probability=True, random_state=42))])
    pipe.fit(x_train, y_train)
    return pipe


def calc_longest(all_df, topk):
    new_all_df = []
    for df in all_df:
        df['target_score'] = df['text'].apply(str).apply(nltk.tokenize.word_tokenize).apply(len)
        new_all_df.append(df)
    # new_all_df = pd.concat(new_all_df)
    calculate_all_score(new_all_df, k=topk)


def calc_lexrank(all_df, train_file_path, topk):
    lexrank = load_lexrank(train_file_path)
    tmp = []
    for df in all_df:
        df['target_score'] = lexrank.rank_sentences([str(v) for v in df['text'].values.tolist()], threshold=0.1)
        tmp.append(df)
    calculate_all_score(tmp, k=topk)


def calc_hardest(all_df, topk):
    tmp = []
    for df in all_df:
        df['target_score'] = [sum([textstat.flesch_reading_ease(str(s)) for s in item]) / len(item) for item in
                              df['text'].apply(str).apply(nltk.sent_tokenize).tolist()]
        tmp.append(df)

    # higher flesch scores means easier to understand
    calculate_all_score(tmp, k=topk, ascending=True)


def calc_random(all_df, topk):
    tmp = []
    for df in all_df:
        r = list(range(len(df)))
        random.shuffle(r)
        df['target_score'] = r
        tmp.append(df)
    calculate_all_score(tmp, k=topk)


def calc_svm(all_df, train_file_path, topk):
    svm_pipeline = load_svm(train_file_path)

    tmp = []
    for df in all_df:
        df['target_score'] = svm_pipeline.predict_proba(df['text'].apply(str).values.tolist())[:, 1]
        tmp.append(df)

    calculate_all_score(tmp, k=topk)


def calc_robert(all_df, topk):
    tmp = []
    for df in all_df:
        df['target_score'] = df['pred']
        tmp.append(df)
    calculate_all_score(tmp, k=topk)


def calc_ground_truth(all_df, topk):
    tmp = []
    for df in all_df:
        df['target_score'] = df.sort_values(by='label', ascending=False).rank()
        tmp.append(df)
    calculate_all_score(tmp, topk)


def main(args):
    all_df = [item[1] for item in pd.read_csv('data/rank_v2.csv').groupby('chapter')]

    if args.method == 'longest':
        calc_longest(all_df, args.topk)
    elif args.method == 'lexrank':
        calc_lexrank(all_df, args.train_file_path, args.topk)
    elif args.method == 'hardest':
        calc_hardest(all_df, args.topk)
    elif args.method == 'random':
        calc_random(all_df, args.topk)
    elif args.method == 'svm':
        calc_svm(all_df, args.train_file_path, args.topk)
    elif args.method == 'robert':
        calc_robert(all_df, args.topk)
    elif args.method == 'ground_truth':
        calc_ground_truth(all_df, args.topk)
    else:

        raise Exception('not implemented ')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='ground_truth')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--train_file_path', type=str, default='../raw_data/qg_train.json', )
    parser.add_argument('--valid_file_path', type=str, default='../raw_data/qg_valid.json', )

    main(parser.parse_args())
