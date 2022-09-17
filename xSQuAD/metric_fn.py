import os
import random
from itertools import chain
from multiprocessing import Pool

import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.util import ngrams
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import recall_score

os.environ["TOKENIZERS_PARALLELISM"] = "false"

random.seed(42)


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


def average_dissimilarity_candidates(c):
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
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def distinct_ngram_score(a_set_of_tokenized_txt, n=1):
    all_possible_ngrams = []
    for item in a_set_of_tokenized_txt:
        all_possible_ngrams.append(ngrams(item, n=n))
    all_possible_ngrams = list(chain(*all_possible_ngrams))
    return len(set(all_possible_ngrams)) / len(all_possible_ngrams)


def eval_all_df_ranking_scores(all_df, k=10, ascending=False):
    avg_recall_10 = []
    avg_map_10 = []
    avg_adc10 = []
    avg_bleu1 = []
    avg_bleu2 = []
    avg_bleu3 = []
    avg_bleu4 = []
    avg_distinct1 = []
    avg_distinct2 = []

    for df in all_df:
        ground_truth = df.sort_values(by='label', ascending=False)['label'].values

        corpus_label = df.sort_values(by='target_score', ascending=ascending)['label'].values

        corpus_text = df.sort_values(by='target_score', ascending=ascending)['text'].values

        res = eval_df_for_ranking(corpus_label, corpus_text, ground_truth, k)
        recall_v, map_v, adc_v, bleu1_v, bleu2_v, bleu3_v, bleu4_v, distinct1, distinct2 = res.values()

        avg_recall_10.append(recall_v)
        avg_map_10.append(map_v)
        avg_adc10.append(adc_v)
        avg_bleu1.append(bleu1_v)
        avg_bleu2.append(bleu2_v)
        avg_bleu3.append(bleu3_v)
        avg_bleu4.append(bleu4_v)
        avg_distinct1.append(distinct1)
        avg_distinct2.append(distinct2)

    avg_recal_v = (sum(avg_recall_10) / len(avg_recall_10)) * 100
    avg_map_v = (sum(avg_map_10) / len(avg_map_10)) * 100
    avg_adc_v = (sum(avg_adc10) / len(avg_adc10)) * 100
    avg_bleu1_v = (sum(avg_bleu1) / len(avg_bleu1)) * 100
    avg_bleu2_v = (sum(avg_bleu2) / len(avg_bleu2)) * 100
    avg_bleu3_v = (sum(avg_bleu3) / len(avg_bleu3)) * 100
    avg_bleu4_v = (sum(avg_bleu4) / len(avg_bleu4)) * 100
    avg_dist1 = (sum(avg_distinct1) / len(avg_distinct1)) * 100
    avg_dist2 = (sum(avg_distinct2) / len(avg_distinct2)) * 100

    return {
        'recall': avg_recal_v,
        'map': avg_map_v,
        'adc': avg_adc_v,
        'avg_self-bleu1': avg_bleu1_v,
        'avg_self-bleu2': avg_bleu2_v,
        'avg_self-bleu3': avg_bleu3_v,
        'avg_self-bleu4': avg_bleu4_v,
        'avg_distinct1': avg_dist1,
        'avg_distinct2': avg_dist2,
    }


def eval_all_df_generation_scores(all_dfs, ground_dfs):
    avg_adc = []
    avg_selfbleu1 = []
    avg_selfbleu2 = []
    avg_selfbleu3 = []
    avg_selfbleu4 = []
    avg_distinct1 = []
    avg_distinct2 = []
    avg_bleu1 = []
    avg_bleu2 = []
    avg_bleu3 = []
    avg_bleu4 = []
    avg_meteor = []

    for rdf, gdf in zip(all_dfs, ground_dfs):
        ground_q = gdf['question'].values
        generated_q = rdf['generation'].values
        res = eval_df_for_generation(ground_q, generated_q)
        # res.values()
        adc, selfbleu1_val, selfbleu2_val, \
        selfbleu3_val, selfbleu4_val, distinc1, \
        distinc2, b_val1, b_val2, b_val3, b_val4, meteor = res.values()

        avg_adc.append(adc)

        avg_selfbleu1.append(selfbleu1_val)
        avg_selfbleu2.append(selfbleu2_val)
        avg_selfbleu3.append(selfbleu3_val)
        avg_selfbleu4.append(selfbleu4_val)

        avg_distinct1.append(distinc1)
        avg_distinct2.append(distinc2)

        avg_bleu1.append(b_val1)
        avg_bleu2.append(b_val2)
        avg_bleu3.append(b_val3)
        avg_bleu4.append(b_val4)

        avg_meteor.append(meteor)

    avg_adc_v = (sum(avg_adc) / len(avg_adc)) * 100

    avg_selfbleu1_v = (sum(avg_selfbleu1) / len(avg_selfbleu1)) * 100
    avg_selfbleu2_v = (sum(avg_selfbleu2) / len(avg_selfbleu2)) * 100
    avg_selfbleu3_v = (sum(avg_selfbleu3) / len(avg_selfbleu3)) * 100
    avg_selfbleu4_v = (sum(avg_selfbleu4) / len(avg_selfbleu4)) * 100

    avg_bleu1_v = (sum(avg_bleu1) / len(avg_bleu1)) * 100
    avg_bleu2_v = (sum(avg_bleu2) / len(avg_bleu2)) * 100
    avg_bleu3_v = (sum(avg_bleu3) / len(avg_bleu3)) * 100
    avg_bleu4_v = (sum(avg_bleu4) / len(avg_bleu4)) * 100

    avg_dist1 = (sum(avg_distinct1) / len(avg_distinct1)) * 100
    avg_dist2 = (sum(avg_distinct2) / len(avg_distinct2)) * 100

    avg_meteor_v = (sum(avg_meteor) / len(avg_meteor)) * 100

    return {
        'adc': avg_adc_v,
        'avg_selfbleu1': avg_selfbleu1_v,
        'avg_selfbleu2': avg_selfbleu2_v,
        'avg_selfbleu3': avg_selfbleu3_v,
        'avg_selfbleu4': avg_selfbleu4_v,

        'avg_bleu1': avg_bleu1_v,
        'avg_bleu2': avg_bleu2_v,
        'avg_bleu3': avg_bleu3_v,
        'avg_bleu4': avg_bleu4_v,

        'avg_distinct1': avg_dist1,
        'avg_distinct2': avg_dist2,

        'avg_meteor': avg_meteor_v,
    }


# def eval_all_metrics_for_generation(corpus_text):
#     adc_val = average_dissimilarity_candidates(corpus_text)
#
#     tokeinized_corpus = [word_tokenize(str(s).lower()) for s in corpus_text]
#
#     bleu1_val = bleu1.get_bleu_parallel(tokeinized_corpus)
#     bleu2_val = bleu2.get_bleu_parallel(tokeinized_corpus)
#     bleu3_val = bleu3.get_bleu_parallel(tokeinized_corpus)
#     bleu4_val = bleu4.get_bleu_parallel(tokeinized_corpus)
#
#     gram1_score = distinct_ngram_score(tokeinized_corpus, 1)
#     gram2_score = distinct_ngram_score(tokeinized_corpus, 2)
#
#     #  evaluate them?
#     res = {
#         'adc': adc_val,
#         'self-blue1': bleu1_val,
#         'self-blue2': bleu2_val,
#         'self-blue3': bleu3_val,
#         'self-blue4': bleu4_val,
#         'distinct1': gram1_score,
#         'distinct2': gram2_score,
#     }
#
#     return res
# def calculate_generation_diversity_scores(all_df):
#     avg_adc10 = []
#     avg_bleu1 = []
#     avg_bleu2 = []
#     avg_bleu3 = []
#     avg_bleu4 = []
#     avg_distinct1 = []
#     avg_distinct2 = []
#
#     for df in all_df:
#         corpus_text = df['question'].values
#
#         res = eval_all_metrics_for_generation(corpus_text)
#         adc_v, bleu1_v, bleu2_v, bleu3_v, bleu4_v, distinct1, distinct2 = res.values()
#
#         avg_adc10.append(adc_v)
#         avg_bleu1.append(bleu1_v)
#         avg_bleu2.append(bleu2_v)
#         avg_bleu3.append(bleu3_v)
#         avg_bleu4.append(bleu4_v)
#         avg_distinct1.append(distinct1)
#         avg_distinct2.append(distinct2)

def eval_df_for_generation(generated, ground_truth):
    adc = average_dissimilarity_candidates(generated)
    tokeinized_corpus = [word_tokenize(str(s).lower()) for s in generated]

    tokenized_references = [word_tokenize(str(s).lower()) for s in ground_truth]

    selfbleu1_val = bleu1.get_bleu_parallel(tokeinized_corpus)
    selfbleu2_val = bleu2.get_bleu_parallel(tokeinized_corpus)
    selfbleu3_val = bleu3.get_bleu_parallel(tokeinized_corpus)
    selfbleu4_val = bleu4.get_bleu_parallel(tokeinized_corpus)

    gram1_score = distinct_ngram_score(tokeinized_corpus, 1)
    gram2_score = distinct_ngram_score(tokeinized_corpus, 2)

    b_val1 = sum([bleu1.calc_bleu(tokenized_references, g, weight=(1, 0, 0, 0)) for g in tokeinized_corpus])
    b_val1 = b_val1 / len(tokeinized_corpus)

    b_val2 = sum([bleu2.calc_bleu(tokenized_references, g, weight=(0.5, 0.5, 0, 0)) for g in tokeinized_corpus])
    b_val2 = b_val2 / len(tokeinized_corpus)

    b_val3 = sum([bleu3.calc_bleu(tokenized_references, g, weight=(1 / 3, 1 / 3, 1 / 3, 0)) for g in tokeinized_corpus])
    b_val3 = b_val3 / len(tokeinized_corpus)

    b_val4 = sum([bleu3.calc_bleu(tokenized_references, g, weight=(0.25, 0.25, 0.25, 0.25)) for g in tokeinized_corpus])
    b_val4 = b_val4 / len(tokeinized_corpus)

    meteor = sum([meteor_score(references=ground_truth, hypothesis=g) for g in generated]) / len(generated)

    res = {
        'adc': adc,
        'self-bleu1': selfbleu1_val,
        'self-bleu2': selfbleu2_val,
        'self-bleu3': selfbleu3_val,
        'self-bleu4': selfbleu4_val,
        'distinct1': gram1_score,
        'distinct2': gram2_score,
        'bleu1_to_org': b_val1,
        'bleu2_to_org': b_val2,
        'bleu3_to_org': b_val3,
        'bleu4_to_org': b_val4,
        'meteor': meteor,
    }

    return res


def eval_df_for_ranking(corpus_label, corpus_text, ground_truth, topk):
    recall_val = recall_score(ground_truth[:topk], corpus_label[:topk])
    map_val = mean_average_precision([corpus_label[:topk]])
    adc_val = average_dissimilarity_candidates(corpus_text[:topk])

    tokeinized_corpus = [word_tokenize(str(s).lower()) for s in corpus_text[:topk]]

    bleu1_val = bleu1.get_bleu_parallel(tokeinized_corpus)
    bleu2_val = bleu2.get_bleu_parallel(tokeinized_corpus)
    bleu3_val = bleu3.get_bleu_parallel(tokeinized_corpus)
    bleu4_val = bleu4.get_bleu_parallel(tokeinized_corpus)

    # do we need a diverse set of questions?
    gram1_score = distinct_ngram_score(tokeinized_corpus, 1)
    gram2_score = distinct_ngram_score(tokeinized_corpus, 2)

    #  evaluate them?
    res = {
        'recall': recall_val,
        'map': map_val,
        'adc': adc_val,
        'self-blue1': bleu1_val,
        'self-blue2': bleu2_val,
        'self-blue3': bleu3_val,
        'self-blue4': bleu4_val,
        'distinct1': gram1_score,
        'distinct2': gram2_score,
    }

    return res
    # return 0, 0, 0, bleu1_val, bleu2_val, bleu3_val, bleu4_val
