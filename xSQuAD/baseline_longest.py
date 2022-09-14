import nltk.tokenize
import pandas as pd
from textstat import textstat
from nltk.tokenize import sent_tokenize
from xSQuAD.utils import all_eval_metric

evaluated_recalls = [10]

all_df = [item[1] for item in pd.read_csv('data/rank_v2.csv').groupby('chapter')]

# avg_recall_5 = []
# avg_prec_5 = []
avg_recall_10 = []
avg_map_10 = []
# avg_adc5 = []
avg_adc10 = []

avg_bleu1 = []
avg_bleu2 = []
avg_bleu3 = []
avg_bleu4 = []

for df in all_df:

    df['longest'] = df['text'].apply(str).apply(nltk.tokenize.word_tokenize).apply(len)

    ground_truth = df.sort_values(by='label', ascending=False)['label'].values

    corpus_label = df.sort_values(by='longest', ascending=False)['label'].values

    corpus_text = df.sort_values(by='longest', ascending=False)['text'].values

    for topk in evaluated_recalls:
        res = all_eval_metric(corpus_label, corpus_text, ground_truth, topk)
        recall_v, map_v, adc_v, bleu1_v, bleu2_v, bleu3_v, bleu4_v = res.values()

        if topk == 10:
            avg_recall_10.append(recall_v)
            avg_map_10.append(map_v)
            avg_adc10.append(adc_v)
            avg_bleu1.append(bleu1_v)
            avg_bleu2.append(bleu2_v)
            avg_bleu3.append(bleu3_v)
            avg_bleu4.append(bleu4_v)

# print('Avg recall@5 : ', (sum(avg_recall_5) / len(avg_recall_5)) * 100)
print('Avg recall@10: ', (sum(avg_recall_10) / len(avg_recall_10)) * 100)
#
# print('Avg MAP@5 : ', (sum(avg_prec_5) / len(avg_prec_5)) * 100)
print('Avg MAP@10: ', (sum(avg_map_10) / len(avg_map_10)) * 100)

# print('Avg ADC@5 : ', (sum(avg_adc5) / len(avg_adc5)) * 100)
print('Avg ADC@10 : ', (sum(avg_adc10) / len(avg_adc10)) * 100)

print('Avg Bleu1@10 : ', (sum(avg_bleu1) / len(avg_bleu1)) * 100)
print('Avg Bleu2@10 : ', (sum(avg_bleu2) / len(avg_bleu2)) * 100)
print('Avg Bleu3@10 : ', (sum(avg_bleu3) / len(avg_bleu3)) * 100)
print('Avg Bleu4@10 : ', (sum(avg_bleu4) / len(avg_bleu4)) * 100)
