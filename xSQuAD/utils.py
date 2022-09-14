import json
import os
from itertools import chain
from multiprocessing import Pool
from random import sample, shuffle

import matplotlib.colors
import matplotlib.pyplot as plt
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import recall_score


class SelfBleu:
    def __init__(self, test_text='', gram=4):
        super().__init__()
        self.name = 'Self-Bleu'
        self.test_data = test_text
        self.gram = gram
        self.sample_size = 500
        self.reference = None
        self.is_first = True

    def get_name(self):
        return self.name

    def get_score(self, is_fast=True, ignore=False):
        if ignore:
            return 0
        if self.is_first:
            self.get_reference()
            self.is_first = False
        if is_fast:
            return self.get_bleu_fast()
        return self.get_bleu_parallel()

    def get_reference(self):
        if self.reference is None:
            reference = list()
            with open(self.test_data) as real_data:
                for text in real_data:
                    text = nltk.word_tokenize(text)
                    reference.append(text)
            self.reference = reference
            return reference
        else:
            return self.reference

    def get_bleu(self):
        ngram = self.gram
        bleu = list()
        reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        with open(self.test_data) as test_data:
            for hypothesis in test_data:
                hypothesis = nltk.word_tokenize(hypothesis)
                bleu.append(nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                                    smoothing_function=SmoothingFunction().method1))
        return sum(bleu) / len(bleu)

    def calc_bleu(self, reference, hypothesis, weight):
        return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                       smoothing_function=SmoothingFunction().method1)

    def get_bleu_fast(self):
        reference = self.get_reference()
        # random.shuffle(reference)
        reference = reference[0:self.sample_size]
        return self.get_bleu_parallel(reference=reference)

    def get_bleu_parallel(self, reference=None):
        ngram = self.gram
        if reference is None:
            reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        pool = Pool(os.cpu_count())
        result = list()
        sentence_num = len(reference)
        for index in range(sentence_num):
            hypothesis = reference[index]
            other = reference[:index] + reference[index + 1:]
            result.append(pool.apply_async(self.calc_bleu, args=(other, hypothesis, weight)))

        score = 0.0
        cnt = 0
        for i in result:
            score += i.get()
            cnt += 1
        pool.close()
        pool.join()
        return score / cnt


embedder = SentenceTransformer('all-MiniLM-L6-v2')
bleu1 = SelfBleu(gram=1)
bleu2 = SelfBleu(gram=2)
bleu3 = SelfBleu(gram=3)
bleu4 = SelfBleu(gram=4)


def read_json_file(file_path):
    with open(file_path) as outfile:
        data = json.load(outfile)
    return data


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


def calculate_dissimilarity(c):
    # c = corpus_[:topk]
    # Normalize the embeddings to unit length
    corpus_embeddings = embedder.encode(c, normalize_embeddings=True)
    # corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    res = util.cos_sim(corpus_embeddings, corpus_embeddings)
    res = 1 - res.data.numpy()
    non_diag = np.tril(res, k=-1) + np.triu(res, k=1)
    sim_scores = np.sum(np.sum(non_diag, axis=-1, keepdims=False)) / (res.shape[0] * (res.shape[0] - 1))
    return sim_scores


# reference: https://gist.github.com/bwhite/3726239
def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def calculate_all_score(all_df, k=10, ascending=False):
    avg_recall_10 = []
    avg_map_10 = []
    avg_adc10 = []
    avg_bleu1 = []
    avg_bleu2 = []
    avg_bleu3 = []
    avg_bleu4 = []
    for df in all_df:
        ground_truth = df.sort_values(by='label', ascending=False)['label'].values

        corpus_label = df.sort_values(by='target_score', ascending=ascending)['label'].values

        corpus_text = df.sort_values(by='target_score', ascending=ascending)['text'].values

        res = all_eval_metric(corpus_label, corpus_text, ground_truth, k)
        recall_v, map_v, adc_v, bleu1_v, bleu2_v, bleu3_v, bleu4_v = res.values()

        avg_recall_10.append(recall_v)
        avg_map_10.append(map_v)
        avg_adc10.append(adc_v)
        avg_bleu1.append(bleu1_v)
        avg_bleu2.append(bleu2_v)
        avg_bleu3.append(bleu3_v)
        avg_bleu4.append(bleu4_v)

    print('Avg recall@10: ', (sum(avg_recall_10) / len(avg_recall_10)) * 100)
    #
    print('Avg MAP@10: ', (sum(avg_map_10) / len(avg_map_10)) * 100)

    print('Avg ADC@10 : ', (sum(avg_adc10) / len(avg_adc10)) * 100)
    print('Avg Bleu1@10 : ', (sum(avg_bleu1) / len(avg_bleu1)) * 100)
    print('Avg Bleu2@10 : ', (sum(avg_bleu2) / len(avg_bleu2)) * 100)
    print('Avg Bleu3@10 : ', (sum(avg_bleu3) / len(avg_bleu3)) * 100)
    print('Avg Bleu4@10 : ', (sum(avg_bleu4) / len(avg_bleu4)) * 100)


def all_eval_metric(corpus_label, corpus_text, ground_truth, topk):
    recall_val = recall_score(ground_truth[:topk], corpus_label[:topk])
    map_val = mean_average_precision([corpus_label[:topk]])
    adc_val = calculate_dissimilarity(corpus_text[:topk])

    tokeinized_corpus = [word_tokenize(str(s).lower()) for s in corpus_text[:topk]]
    bleu1_val = bleu1.get_bleu_parallel(tokeinized_corpus)
    bleu2_val = bleu2.get_bleu_parallel(tokeinized_corpus)
    bleu3_val = bleu3.get_bleu_parallel(tokeinized_corpus)
    bleu4_val = bleu4.get_bleu_parallel(tokeinized_corpus)
    res = {
        'recall': recall_val,
        'map': map_val,
        'adc': adc_val,
        'blue1': bleu1_val,
        'blue2': bleu2_val,
        'blue3': bleu3_val,
        'blue4': bleu4_val,
    }

    return res
    # return 0, 0, 0, bleu1_val, bleu2_val, bleu3_val, bleu4_val


def categorical_cmap(nc, nsc, cmap="tab10", continuous=False):
    if nc > plt.get_cmap(cmap).N:
        raise ValueError("Too many categories for colormap.")
    if continuous:
        ccolors = plt.get_cmap(cmap)(np.linspace(0, 1, nc))
    else:
        ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
    cols = np.zeros((nc * nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = matplotlib.colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv, nsc).reshape(nsc, 3)
        arhsv[:, 1] = np.linspace(chsv[1], 0.25, nsc)
        arhsv[:, 2] = np.linspace(chsv[2], 1, nsc)
        rgb = matplotlib.colors.hsv_to_rgb(arhsv)
        cols[i * nsc:(i + 1) * nsc, :] = rgb
    cmap = matplotlib.colors.ListedColormap(cols)
    return cmap
