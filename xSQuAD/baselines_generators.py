import argparse
import os

import pandas as pd
from helper_fn import save_result_on_disk
from tqdm import tqdm
from xSQuAD.metric_fn import eval_all_df_generation_scores

from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained('hadifar/tqa_qg_agno')
tokenizer = T5Tokenizer.from_pretrained('hadifar/tqa_qg_agno')


# hadifar/openstax_qg_agno
# hadifar/tqa_qg_agno

def load_selected_df_ranking(args):
    method_file = os.path.join(args.cache_path, args.dataset, args.method, args.method + '.rank.csv')

    ground_df = [item[1] for item in pd.read_csv(args.qg_pairs).groupby('chapter')]
    grnd_len = [len(item) for item in ground_df]

    method_grp = [t[1] for t in pd.read_csv(method_file).groupby('chapter')]
    method_dfs = [item.sort_values(by='target_score', ascending=False)[:l] for item, l in zip(method_grp, grnd_len)]
    return method_dfs, ground_df


def do_generation_and_eval(args):
    tmp = []
    ranked_dfs, ground_dfs = load_selected_df_ranking(args)
    for rdf, gdf in tqdm(zip(ranked_dfs, ground_dfs)):
        if len(rdf) < 2: # we dropped chapter with 1 since we cannot report selfbleu & distinct
            continue

        gen_df = []
        for ctx in rdf['text'].apply(str).tolist():
            ctx = "context : {}".format(ctx)
            input_ids = tokenizer(ctx, return_tensors="pt").input_ids
            outputs = model.generate(input_ids)
            tokenized_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            gen_df.append(tokenized_output)
        rdf['generated_by_model'] = gen_df
        rdf['ground_truth'] = gdf['question'].tolist()
        tmp.append(rdf)

    result_dic = eval_all_df_generation_scores(tmp, ground_dfs)
    save_result_on_disk(tmp, result_dic, args.cache_path, args.dataset, args.method, args.task)


def main(args):
    do_generation_and_eval(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='random')
    parser.add_argument('--task', type=str, default='generation')
    parser.add_argument('--dataset', type=str, default='tqa')
    parser.add_argument('--qg_pairs', type=str, default='../raw_data/tqa/qg_ground_question_context_pairs.csv')
    # parser.add_argument('--train_file_path', type=str, default='../raw_data/qg_train.json', )
    # parser.add_argument('--valid_file_path', type=str, default='../raw_data/qg_valid.json', )
    # parser.add_argument('--ranking_file_path', type=str, default='data/rank_v3.csv', )
    parser.add_argument('--cache_path', type=str, default='../xSQuAD/cached/', )
    args = parser.parse_args()
    # for v in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.01, 0.001, 0.0001, 0.05, 0.005, 0.0005, 0.09,
    #           0.009, 0.0009]:
    #     args.method = "topicwise_l{}".format(v)
    main(args)
