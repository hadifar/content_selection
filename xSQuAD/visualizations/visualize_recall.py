import json

import pandas as pd
import matplotlib.pyplot as plt
from lexrank import LexRank, STOPWORDS

evaluated_recalls = [1, 3, 5, 10, 15]
yaxis = [0.0, 0.25, 0.5, 0.75, 1]

with open('../../raw_data/openstax/qg_valid.json') as inpfile:
    dataet = json.load(inpfile)
    lexrank = LexRank([item['intro'] + ' ' + item['chapter_text'] for item in dataet], stopwords=STOPWORDS['en'])

df = [item[1] for item in pd.read_csv('../data/rank_v2.csv').groupby('chapter')][0]
lex_recall = []
ground_truth = df.sort_values(by='label', ascending=False)['label'].values
df['lexrank'] = lexrank.rank_sentences(df['text'].values.tolist(), threshold=0.1)
df = df.sort_values(by='lexrank', ascending=False)
print('lex rank')
for topk in evaluated_recalls:
    corpus = [item for item in df['label'].values[:topk].tolist()]
    recal_val = sum(corpus) / sum(ground_truth[:topk])
    lex_recall.append(recal_val)
    print('recall@{} --> {}'.format(topk, recal_val))

print('-' * 50)

print('random')
df = [item[1] for item in pd.read_csv('../data/rank_v2.csv').groupby('chapter')][0]
randm_recall = []
corpus = df['label'].sample(max(evaluated_recalls), random_state=42).values.tolist()
for topk in evaluated_recalls:
    recall_val = sum(corpus[:topk]) / sum(ground_truth[:topk])
    randm_recall.append(recall_val)
    print('recall@{} --> {}'.format(topk, recall_val))

print('-' * 50)
df = pd.read_csv('../data/sim_0.1/rank_topic_0.9_full.csv')
df = df.sort_values(by='score', ascending=False)
robert_recall = []
ground_truth = df.sort_values(by='label', ascending=False)['label'].values
for topk in evaluated_recalls:
    corpus = [item for item in df['label'].values[:topk].tolist()]
    recal_val = sum(corpus) / sum(ground_truth[:topk])
    robert_recall.append(recal_val)
    print('recall@{} --> {}'.format(topk, recal_val))
    # Normalize the embeddings to unit length

print('-' * 50)
print('MMR-based ranking')
mrr_recall = []
df = pd.read_csv('../example/rank_topic_0.9_full.csv')
df = df.sort_values(by='mmr', ascending=False)
for topk in evaluated_recalls:
    corpus = [item for item in df['label'].values[:topk].tolist()]
    recal_val = sum(corpus) / sum(ground_truth[:topk])
    mrr_recall.append(recal_val)
    print('recall@{} --> {}'.format(topk, recal_val))

plt.plot(range(len(evaluated_recalls)), randm_recall, label="Random", marker='v')
plt.plot(range(len(evaluated_recalls)), lex_recall, label="LexRank", marker='<')
plt.plot(range(len(evaluated_recalls)), robert_recall, label="PairWise", marker='>')
plt.plot(range(len(evaluated_recalls)), mrr_recall, label="TopicWise", marker='^')
plt.legend()
plt.xticks(ticks=range(len(evaluated_recalls)), labels=[str(i) for i in evaluated_recalls])
plt.yticks(ticks=yaxis, labels=[str(i) for i in yaxis])

plt.xlabel('K')
plt.ylabel('Recall')

plt.show()
