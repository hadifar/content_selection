import argparse
import json
import os

import pandas as pd
from helper_fn import save_result_on_disk

from xSQuAD.metric_fn import eval_all_df_generation_scores

from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained('hadifar/openstax_qg_agno')
tokenizer = T5Tokenizer.from_pretrained('hadifar/openstax_qg_agno')


def calc_ground_truth_gen(args):
    tmp = []
    ranked_dfs, ground_dfs = load_selected_df_ranking(args)
    for rdf, gdf in zip(ranked_dfs,ground_dfs):
        gen_df = []
        for ctx in rdf['text'].tolist():
            input_ids = tokenizer(ctx, return_tensors="pt").input_ids
            outputs = model.generate(input_ids)
            tokenized_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            gen_df.append(tokenized_output)
        rdf['generation'] = gen_df
        rdf['ground_truth'] = gdf['question'].tolist()
        tmp.append(rdf)

    result_dic = eval_all_df_generation_scores(tmp, ground_dfs)
    save_result_on_disk(tmp, result_dic, args.cache_path, args.method, args.task)


def load_selected_df_ranking(args):
    method_file = os.path.join(args.cache_path, args.method, args.method + '.rank.csv')

    ground_df = [item[1] for item in pd.read_csv(args.qg_pairs).groupby('chapter')]
    grnd_len = [len(item) for item in ground_df]

    method_grp = pd.read_csv(method_file).groupby('chapter')
    method_dfs = [item[1].sort_values(by='target_score', ascending=False)[:l] for item, l in zip(method_grp, grnd_len)]
    return method_dfs[:1], ground_df[:1]


# def load_ground_df_generation(args):
#     return [item[1] for item in pd.read_csv(args.qg_pairs).groupby('chapter')]
#

# def calc_roberta_gen(args):
#     from transformers import T5ForConditionalGeneration, T5Tokenizer
#     model = T5ForConditionalGeneration.from_pretrained('hadifar/openstax_qg_agno')
#     tokenizer = T5Tokenizer.from_pretrained('hadifar/openstax_qg_agno')
#
#     tmp = []
#     # todo: for debug purpose
#     selected_df = load_selected_df_ranking(args)[:1]
#     for df in selected_df:
#         gen_df = []
#         for ctx in df['text'].tolist():
#             input_ids = tokenizer(ctx, return_tensors="pt").input_ids
#             outputs = model.generate(input_ids)
#             tokenized_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
#             gen_df.append(tokenized_output)
#         df['generation'] = gen_df
#         tmp.append(df)
#
#     result_dic = eval_all_df_generation_scores(tmp)
#     save_result_on_disk(tmp, result_dic, args.cache_path, args.method, args.task)


def main(args):
    if args.method == 'ground_truth':
        calc_ground_truth_gen(args)
    elif args.method == 'roberta':
        pass
        # calc_roberta_gen(args)

    # elif args.method == 'longest':
    #     calc_longest(all_df, args.topk)
    # elif args.method == 'lexrank':
    #     calc_lexrank(all_df, args.train_file_path, args.topk)
    # elif args.method == 'hardest':
    #     calc_hardest(all_df, args.topk)
    # elif args.method == 'random':
    #     calc_random(all_df, args.topk)
    # elif args.method == 'svm':
    #     calc_svm(all_df, args.train_file_path, args.topk)
    # elif args.method == 'robert':
    #     calc_robert(all_df, args.topk)

    else:
        raise Exception('not implemented ')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='ground_truth')
    parser.add_argument('--task', type=str, default='generation')
    parser.add_argument('--qg_pairs', type=str, default='raw_data/qg_ground_question_context_pairs.csv')
    # parser.add_argument('--train_file_path', type=str, default='../raw_data/qg_train.json', )
    # parser.add_argument('--valid_file_path', type=str, default='../raw_data/qg_valid.json', )
    # parser.add_argument('--ranking_file_path', type=str, default='data/rank_v3.csv', )
    parser.add_argument('--cache_path', type=str, default='xSQuAD/cached/', )

    main(parser.parse_args())
