import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import recall_score, precision_score, average_precision_score

from xSQuAD.utils import calculate_dissimilarity,mapk

evaluated_recalls = [5, 10]

all_df = [item[1] for item in pd.read_csv('data/rank_v2.csv').groupby('chapter')]

avg_recall_5 = []
avg_prec_5 = []
avg_recall_10 = []
avg_prec_10 = []
avg_adc5 = []

for df in all_df:
    ground_truth = df.sort_values(by='label', ascending=False)['label'].values

    corpus_label = df['label'].sample(max(evaluated_recalls), random_state=4).values.tolist()

    corpus_text = df['text'].sample(max(evaluated_recalls), random_state=4).values.tolist()

    for topk in evaluated_recalls:
        recall_val = recall_score(ground_truth[:topk], corpus_label[:topk])
        prec_val = precision_score(ground_truth[:topk], corpus_label[:topk], average='micro', zero_division=0)

        # print(prec_val)
        # recall_val = sum(corpus[:topk]) / sum(ground_truth[:topk])
        if topk == 5:
            avg_adc5.append(calculate_dissimilarity(corpus_text[:topk]))
            avg_recall_5.append(recall_val)
            avg_prec_5.append(prec_val)
        elif topk == 10:
            avg_recall_10.append(recall_val)
            avg_prec_10.append(prec_val)
        else:
            raise Exception('k value is not defined...')

print('Avg recall@5 : ', (sum(avg_recall_5) / len(avg_recall_5)) * 100)
print('Avg recall@10: ', (sum(avg_recall_10) / len(avg_recall_10)) * 100)
#
print('Avg Prec@5 : ', (sum(avg_prec_5) / len(avg_prec_5)) * 100)
print('Avg Prec@10: ', (sum(avg_prec_10) / len(avg_prec_10)) * 100)


print('Avg ADC@5 : ', (sum(avg_adc5) / len(avg_adc5)) * 100)