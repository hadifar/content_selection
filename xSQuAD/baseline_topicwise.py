import pandas as pd

from xSQuAD.metric_fn import eval_df_for_ranking

evaluated_recalls = [10]
import json
import numpy as np

for value in [0.1]:
    print('lambda value: {}'.format(value))
    all_dfs = [item[1] for item in pd.read_csv('data/rank_v2.csv').groupby('chapter')]
    topic_sorted_dfs = [item[1] for item in pd.read_csv('data/rank_topic_{}_full.csv'.format(value)).groupby('chapter')]

    # avg_recall_20 = []
    # avg_prec_20 = []
    avg_recall_10 = []
    avg_map_10 = []
    # avg_adc20 = []
    avg_adc10 = []
    avg_bleu1 = []
    avg_bleu2 = []
    avg_bleu3 = []
    avg_bleu4 = []

    for df, df2 in zip(all_dfs, topic_sorted_dfs):

        ground_truth = df.sort_values(by='label', ascending=False)['label'].values

        df2 = df2.sort_values(by='mmr', ascending=False)
        corpus_label = df2['label'].values
        corpus_text = df2['text'].values

        # print(df2['cluster_prob'].values)
        for topk in evaluated_recalls:
            res = eval_df_for_ranking(corpus_label, corpus_text,
                                      ground_truth, topk)
            recall_v, map_v, adc_v, bleu1_v, bleu2_v, bleu3_v, bleu4_v = res.values()
            if topk == 10:
                avg_recall_10.append(recall_v)
                avg_map_10.append(map_v)
                avg_adc10.append(adc_v)
                avg_bleu1.append(bleu1_v)
                avg_bleu2.append(bleu2_v)
                avg_bleu3.append(bleu3_v)
                avg_bleu4.append(bleu4_v)

            else:
                raise Exception('k value is not defined...')

    print('Avg recall@10: ', (sum(avg_recall_10) / len(avg_recall_10)) * 100)
    # print('Avg recall@20 : ', (sum(avg_recall_20) / len(avg_recall_20)) * 100)
    #
    print('Avg Prec@10: ', (sum(avg_map_10) / len(avg_map_10)) * 100)
    # print('Avg Prec@20 : ', (sum(avg_prec_20) / len(avg_prec_20)) * 100)
    #
    print('Avg ADC@10 : ', (sum(avg_adc10) / len(avg_adc10)) * 100)

    print('Avg Bleu1@10 : ', (sum(avg_bleu1) / len(avg_bleu1)) * 100)
    print('Avg Bleu2@10 : ', (sum(avg_bleu2) / len(avg_bleu2)) * 100)
    print('Avg Bleu3@10 : ', (sum(avg_bleu3) / len(avg_bleu3)) * 100)
    print('Avg Bleu4@10 : ', (sum(avg_bleu4) / len(avg_bleu4)) * 100)

    # print('Avg ADC@20 : ', (sum(avg_adc20) / len(avg_adc20)) * 100)
    print('-' * 100)
