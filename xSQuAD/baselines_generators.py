import argparse
import os

import pandas as pd


def calc_ground_truth_gen(selected_df):
    pass

def calc_roberta_gen(selected_df):
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    model = T5ForConditionalGeneration.from_pretrained('hadifar/openstax_qg_agno')
    tokenizer = T5Tokenizer.from_pretrained('hadifar/openstax_qg_agno')

    all_gen = []
    # todo: for debug purpose
    selected_df = selected_df[:1]
    for df in selected_df[:1]:
        for ctx in df['text'].tolist():
            input_ids = tokenizer(ctx, return_tensors="pt").input_ids
            outputs = model.generate(input_ids)
            tokenized_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            all_gen.append(tokenized_output)

    result_dic = calculate_ranking_scores(tmp, args.topk)
    store_ranking(tmp, result_dic, args)

    selected_df = pd.concat(selected_df)
    selected_df['question'] = all_gen


def main(args):
    ground_file = os.path.join(args.cache_path, 'ground_truth', 'ground_truth.csv')
    method_file = os.path.join(args.cache_path, args.method, args.method + '.csv')

    grnd_grp = pd.read_csv(ground_file).groupby('chapter')
    grnd_len = [len([subitem[1] for subitem in item[1].groupby('label')][1]) for item in grnd_grp]

    method_grp = pd.read_csv(method_file).groupby('chapter')
    method_dfs = [item[1].sort_values(by='target_score',ascending=False)[:l] for item, l in zip(method_grp, grnd_len)]

    if args.method == 'ground_truth':
        calc_ground_truth_gen(method_dfs)
    elif args.method == 'roberta':
        calc_roberta_gen(method_dfs)

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
    parser.add_argument('--method', type=str, default='roberta')
    parser.add_argument('--ground_truth', type=str, default='xSQuAD/cached/ground_truth/ground_truth.csv')
    # parser.add_argument('--train_file_path', type=str, default='../raw_data/qg_train.json', )
    # parser.add_argument('--valid_file_path', type=str, default='../raw_data/qg_valid.json', )
    # parser.add_argument('--ranking_file_path', type=str, default='data/rank_v3.csv', )
    parser.add_argument('--cache_path', type=str, default='xSQuAD/cached/', )

    main(parser.parse_args())
