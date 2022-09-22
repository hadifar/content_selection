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
from tqdm import tqdm

from xSQuAD.helper_fn import read_json_file, extract_paragraph_pointwise, save_result_on_disk, cache_exist, \
    pretty_print_results
from xSQuAD.metric_fn import eval_all_df_ranking_scores


#
# def store_ranking(all_df, result_dic, args):
#     folder_dir = os.path.join(args.cache_path, args.method)
#     print('save file in {}'.format(folder_dir))
#     df = pd.concat(all_df)
#     if not os.path.isdir(folder_dir):
#         os.makedirs(folder_dir)
#
#     cache_file = os.path.join(folder_dir, args.method + '.{}.csv'.format(args.task))
#     res_file = os.path.join(folder_dir, args.method + '.{}.json'.format(args.task))
#     # save rankings
#     df.to_csv(cache_file, index=False)
#     # save results
#     json.dump(result_dic, open(res_file, 'w'), indent=4)


def load_lexrank(train_file_path):
    with open(train_file_path) as inpfile:
        raw_dataset = json.load(inpfile)
        lexrank = LexRank([item['intro'] + ' ' + item['chapter_text'] for item in raw_dataset],
                          stopwords=STOPWORDS['en'])
        return lexrank


def load_svm(train_file_path):
    train_data = extract_paragraph_pointwise(read_json_file(train_file_path), 'train', negative_samples_weights=2)
    # valid_data = extract_paragraph_pointwise(read_json_file('raw_data/qg_valid.json')[:2], 'valid')
    x_train, y_train = [str(item[0]) for item in train_data], [item[1] for item in train_data]
    # x_valid, y_valid = [str(item[0]) for item in train_data], [item[1] for item in train_data]
    pipe = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 2))), ('svc', SVC(probability=True, random_state=42))])
    pipe.fit(x_train, y_train)
    return pipe


def calc_longest(all_df, args):
    tmp = cache_exist(args.cache_path, args.method, args.task)
    if len(tmp) == 0:
        for df in all_df:
            df['target_score'] = df['text'].apply(str).apply(nltk.tokenize.word_tokenize).apply(len)
            tmp.append(df)

        result_dic = eval_all_df_ranking_scores(tmp, k=args.topk)
        save_result_on_disk(tmp, result_dic, args.cache_path, args.method, args.task)

    pretty_print_results(args.cache_path, args.method, args.task)


def calc_lexrank(all_df, args):
    tmp = cache_exist(args.cache_path, args.method, args.task)
    if len(tmp) == 0:
        lexrank = load_lexrank(args.train_file_path)
        for df in all_df:
            df['target_score'] = lexrank.rank_sentences([str(v) for v in df['text'].values.tolist()], threshold=0.2)
            tmp.append(df)
        result_dic = eval_all_df_ranking_scores(tmp, k=args.topk)
        save_result_on_disk(tmp, result_dic, args.cache_path, args.method, args.task)
    pretty_print_results(args.cache_path, args.method, args.task)


def calc_hardest(all_df, args):
    tmp = cache_exist(args.cache_path, args.method, args.task)
    if len(tmp) == 0:
        for df in all_df:
            df['target_score'] = [sum([textstat.flesch_reading_ease(str(s)) for s in item]) / len(item) for item in
                                  df['text'].apply(str).apply(nltk.sent_tokenize).tolist()]
            tmp.append(df)

        # higher flesch scores means easier to understand
        result_dic = eval_all_df_ranking_scores(tmp, k=args.topk, ascending=True)
        save_result_on_disk(tmp, result_dic, args.cache_path, args.method, args.task)
    pretty_print_results(args.cache_path, args.method, args.task)


def calc_random(all_df, args):
    tmp = cache_exist(args.cache_path, args.method, args.task)
    if len(tmp) == 0:
        for df in all_df:
            r = list(range(len(df)))
            random.shuffle(r)
            df['target_score'] = r
            tmp.append(df)

        result_dic = eval_all_df_ranking_scores(tmp, k=args.topk)
        save_result_on_disk(tmp, result_dic, args.cache_path, args.method, args.task)

    pretty_print_results(args.cache_path, args.method, args.task)


def calc_svm(all_df, args):
    tmp = cache_exist(args.cache_path, args.method, args.task)
    if len(tmp) == 0:
        svm_pipeline = load_svm(args.train_file_path)
        for df in all_df:
            df['target_score'] = svm_pipeline.predict_proba(df['text'].apply(str).values.tolist())[:, 1]
            tmp.append(df)

        result_dic = eval_all_df_ranking_scores(tmp, k=args.topk)
        save_result_on_disk(tmp, result_dic, args.cache_path, args.method, args.task)

    pretty_print_results(args.cache_path, args.method, args.task)


def calc_robert(all_df, args):
    tmp = cache_exist(args.cache_path, args.method, args.task)
    if len(tmp) == 0:
        for df in all_df:
            df['target_score'] = df['score']
            tmp.append(df)
        result_dic = eval_all_df_ranking_scores(tmp, k=args.topk)
        save_result_on_disk(tmp, result_dic, args.cache_path, args.method, args.task)
    pretty_print_results(args.cache_path, args.method, args.task)


def calc_ground_truth(all_df, args):
    tmp = cache_exist(args.cache_path, args.method, args.task)
    if len(tmp) == 0:
        for df in tqdm(all_df):
            df['target_score'] = df.sort_values(by='label', ascending=False).rank()['label']
            tmp.append(df)
        result_dic = eval_all_df_ranking_scores(tmp, args.topk)
        save_result_on_disk(tmp, result_dic, args.cache_path, args.method, args.task)
    pretty_print_results(args.cache_path, args.method, args.task)


def calc_topicwise(all_df, args):
    args.method = "{}_l{}".format(args.method, args.lambda_score)
    tmp = cache_exist(args.cache_path, args.method, args.task)
    if len(tmp) == 0:
        for df in all_df:
            df['target_score'] = df['mmr']
            tmp.append(df)
        result_dic = eval_all_df_ranking_scores(tmp, k=args.topk)
        save_result_on_disk(tmp, result_dic, args.cache_path, args.method, args.task)
    pretty_print_results(args.cache_path, args.method, args.task)


def load_df_files(ranking_file_path):
    all_df = [item[1] for item in pd.read_csv(ranking_file_path).groupby('chapter')]
    return all_df


def main(args):
    args.ranking_file_path = 'data/rank_topic_based_{}_len25.csv'.format(args.lambda_score)
    all_df = load_df_files(args.ranking_file_path)
    print('method of choice : {}'.format(args.method))

    if args.method == 'longest':
        calc_longest(all_df, args)
    elif args.method == 'lexrank':
        calc_lexrank(all_df, args)
    elif args.method == 'hardest':
        calc_hardest(all_df, args)
    elif args.method == 'random':
        calc_random(all_df, args)
    elif args.method == 'svm':
        calc_svm(all_df, args)
    elif args.method == 'roberta':
        calc_robert(all_df, args)
    elif args.method == 'topicwise':
        calc_topicwise(all_df, args)
    elif args.method == 'ground_truth':
        calc_ground_truth(all_df, args)
    else:
        raise Exception('not implemented ')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='topicwise')
    parser.add_argument('--task', type=str, default='rank')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--lambda_score', type=float, default=1.0)
    parser.add_argument('--train_file_path', type=str, default='../raw_data/qg_train.json', )
    parser.add_argument('--valid_file_path', type=str, default='../raw_data/qg_valid.json', )
    # parser.add_argument('--ranking_file_path', type=str, default='data/rank_topic_based_0.0.csv')
    parser.add_argument('--cache_path', type=str, default='cached/')
    args = parser.parse_args()

    for v in [0.01,0.001,0.0001,0.05,0.005,0.0005,0.09,0.009,0.0009]:
        args.lambda_score = v
        args.method = 'topicwise'
        main(args)
