import json

import pandas as pd
import numpy as np
from lexrank import LexRank, STOPWORDS
from sentence_transformers import SentenceTransformer, util

embedder = SentenceTransformer('all-MiniLM-L6-v2')

evaluated_ADC = [2, 4, 6, 8, 10]
yaxis = [0.0, 0.25, 0.5, 0.75, 1]

print('random')
df = [item[1] for item in pd.read_csv('../data/rank_v2.csv').groupby('chapter')][0]
corpus_ = df['text'].sample(max(evaluated_ADC), random_state=42).values.tolist()
for topk in evaluated_ADC:
    corpus = corpus_[:topk]
    # Normalize the embeddings to unit length
    corpus_embeddings = embedder.encode(corpus)
    corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    res = util.cos_sim(corpus_embeddings, corpus_embeddings)
    res = 1 - res.data.numpy()
    non_diag = np.tril(res, k=-1) + np.triu(res, k=1)
    sim_scores = np.sum(np.sum(non_diag, axis=-1) / res.shape[0]) / res.shape[0]
    print('topk: {} --> avg-dist-to-other-candidate: {}'.format(topk, sim_scores))

with open('../data/qg_valid.json') as inpfile:
    dataet = json.load(inpfile)
    lexrank = LexRank([item['intro'] + ' ' + item['chapter_text'] for item in dataet], stopwords=STOPWORDS['en'])

df = [item[1] for item in pd.read_csv('../data/rank_v2.csv').groupby('chapter')][0]
lex_recall = []
df['lexrank'] = lexrank.rank_sentences(df['text'].values.tolist(), threshold=0.1)
df = df.sort_values(by='lexrank', ascending=False)
print('lex rank')
for topk in evaluated_ADC:
    corpus = [item for item in df['text'].values[:topk].tolist()]
    corpus_embeddings = embedder.encode(corpus)
    corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    res = util.cos_sim(corpus_embeddings, corpus_embeddings)
    res = 1 - res.data.numpy()
    non_diag = np.tril(res, k=-1) + np.triu(res, k=1)
    sim_scores = np.sum(np.sum(non_diag, axis=-1) / res.shape[0]) / res.shape[0]
    print('topk: {} --> avg-dist-to-other-candidate: {}'.format(topk, sim_scores))

print('-' * 50)

df = pd.read_csv('../example/rank_topic_0.9_full.csv')
df = df.sort_values(by='score', ascending=False)
for topk in evaluated_ADC:
    corpus = [item for item in df['text'].values[:topk].tolist()]
    # Normalize the embeddings to unit length
    corpus_embeddings = embedder.encode(corpus)
    corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    res = util.cos_sim(corpus_embeddings, corpus_embeddings)
    res = 1 - res.data.numpy()
    non_diag = np.tril(res, k=-1) + np.triu(res, k=1)
    sim_scores = np.sum(np.sum(non_diag, axis=-1) / res.shape[0]) / res.shape[0]
    print('topk: {} --> avg-dist-to-other-candidate: {}'.format(topk, sim_scores))

print('-' * 50)
print('MMR-based ranking')
df = pd.read_csv('../example/rank_topic_0.9_full.csv')
df = df.sort_values(by='mmr', ascending=False)
for topk in evaluated_ADC:
    corpus = [item for item in df['text'].values[:topk].tolist()]
    # Normalize the embeddings to unit length
    corpus_embeddings = embedder.encode(corpus)
    corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    res = util.cos_sim(corpus_embeddings, corpus_embeddings)
    res = 1 - res.data.numpy()
    non_diag = np.tril(res, k=-1) + np.triu(res, k=1)
    sim_scores = np.sum(np.sum(non_diag, axis=-1) / res.shape[0]) / res.shape[0]
    print('topk: {} --> avg-dist-to-other-candidate: {}'.format(topk, sim_scores))
