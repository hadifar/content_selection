import random
from itertools import chain
from random import sample, shuffle


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from data_helper import read_json_file
import pandas as pd

from xSQuAD.utils import all_eval_metric

random.seed(42)


def extract_paragraph_pointwise(data, split):
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
            neg_txt = sample(neg_txt, len(pos_txt))
        all_data.append(pos_txt)
        all_data.append(neg_txt)
    all_data = list(chain(*all_data))
    if split == 'train':
        shuffle(all_data)
    return all_data


train_data = extract_paragraph_pointwise(read_json_file('raw_data/qg_train.json'), 'train')
# valid_data = extract_paragraph_pointwise(read_json_file('raw_data/qg_valid.json')[:2], 'valid')
x_train, y_train = [str(item[0]) for item in train_data], [item[1] for item in train_data]
# x_valid, y_valid = [str(item[0]) for item in train_data], [item[1] for item in train_data]
pipe = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 2))), ('svc', SVC(probability=True,random_state=42))])
pipe.fit(x_train, y_train)

all_df = [item[1] for item in pd.read_csv('xSQuAD/data/rank_v2.csv').groupby('chapter')]

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

evaluated_recalls = [10]
for df in all_df:

    df['svm'] = pipe.predict_proba(df['text'].apply(str).values.tolist())[:, 1]

    ground_truth = df.sort_values(by='label', ascending=False)['label'].values

    corpus_label = df.sort_values(by='svm', ascending=False)['label'].values

    corpus_text = df.sort_values(by='svm', ascending=False)['text'].values

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
