import argparse
import random

import nltk.tokenize
import pandas as pd
from lexrank import LexRank
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import recall_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from textstat import textstat
from tqdm import tqdm

random.seed(42)

from xSQuAD.helper_fn import read_json_file, extract_paragraph_pointwise, save_result_on_disk, cache_exist, \
    pretty_print_results
from xSQuAD.metric_fn import eval_all_df_ranking_scores, eval_all_df_generation_scores, mean_average_precision


def calculate_recall(g):
    ground_truth = g.sort_values('label', ascending=False)[:10].label.tolist()
    corpus_label = g.sort_values('pred_score', ascending=False)[:10].label
    recall_val = recall_score(ground_truth, corpus_label)
    map_val = mean_average_precision([corpus_label])
    return recall_val, map_val


def load_df_files(args):
    df = pd.read_csv(args.valid_file_path)
    ground_dfs = []
    tmp = []
    map_avg = []
    recall_avg = []
    for _, g in df.groupby('chapter'):
        r, m = calculate_recall(g)
        map_avg.append(m)
        recall_avg.append(r)
        ground_questions = g[g['label'] == 1]
        ground_questions = ground_questions.rename(columns={'questions': 'question'})
        generated_questions = g[g['label'] == 0].sort_values(by='pred_score', ascending=False)
        sub_df = generated_questions[:len(ground_questions)]
        sub_df = sub_df.rename(columns={'questions': 'generated_by_model'})
        ground_dfs.append(ground_questions)
        tmp.append(sub_df)

    result_dic = eval_all_df_generation_scores(tmp, ground_dfs)

    for item in result_dic.items():
        print(item)

    print('-' * 100)

    print('recall', sum(recall_avg) / len(recall_avg))
    print('map', sum(map_avg) / len(map_avg))


def main(args):
    all_df = load_df_files(args)
    print('method of choice : {}'.format(args.method))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='topicwise')
    parser.add_argument('--dataset', type=str, default='openstax', choices=['tqa', 'openstax'])
    # parser.add_argument('--task', type=str, default='rank')
    # parser.add_argument('--topk', type=int, default=10)
    # parser.add_argument('--lambda_score', type=float, default=1.0)
    parser.add_argument('--train_file_path', type=str, default='../raw_data/openstax/overgenerate_train.csv')
    parser.add_argument('--valid_file_path', type=str, default='../raw_data/openstax/overgenerate_valid2.csv')

    # parser.add_argument('--ranking_file_path', type=str, default='i')
    parser.add_argument('--cache_path', type=str, default='cached/')
    args = parser.parse_args()
    #
    # for v in [0.0, 0.001, 0.01, 0.5, 0.9, 1.0]:
    #     args.lambda_score = v
    #     args.method = 'topicwise'
    main(args)
