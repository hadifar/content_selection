import json

import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse


def topic_relevance(doc_current, docs_selected, initial_ranking, topic_distribution_matrix, topic_frequency):
    n_examples, _ = topic_distribution_matrix.shape

    final_score = 0

    curr_doc_index = initial_ranking['doc'] == doc_current
    curr_doc_rank_score = initial_ranking[curr_doc_index]['score'].tolist()[0]
    curr_doc_topic_scores = topic_distribution_matrix[initial_ranking.index[curr_doc_index].tolist()[0]]

    for tf in topic_frequency:
        topic, freq = tf
        topic_weight = (freq / n_examples)

        topic_importance_for_curr_doc = topic_weight * curr_doc_rank_score * curr_doc_topic_scores[topic]

        product_score_for_selected = 1
        for s in docs_selected:
            selected_doc_index = initial_ranking['doc'] == s['doc']
            selected_doc_rank_score = initial_ranking[selected_doc_index]['score'].tolist()[0]
            selected_doc_topic_scores = topic_distribution_matrix[initial_ranking.index[selected_doc_index].tolist()[0]]

            product_score_for_selected = product_score_for_selected * (
                    1 - selected_doc_rank_score * selected_doc_topic_scores[topic])

        # tmp_val =
        final_score += (topic_importance_for_curr_doc * product_score_for_selected)
    return curr_doc_rank_score, final_score
    # topic_score_per_para[topic]
    # sim = 0
    # for s in docs_selected:
    #     sim_current = _lookup_sim(
    #         doc_current, s['doc'],
    #         sim_matrix,
    #         initial_ranking)
    #
    #     if sim_current > sim:
    #         sim = sim_current
    #     else:
    #         continue
    # return sim


def _mmr(lambda_score, doc_current, docs_unranked, docs_selected, initial_ranking, topic_score_matrix):
    """Compute mmr"""
    mmr = -1
    doc = None
    topic_frequency = np.asarray(np.unique(topic_score_matrix.argmax(-1), return_counts=True)).T
    for d in docs_unranked:
        # argmax Sim(d_i, d_j)
        first_score, second_score = topic_relevance(doc_current,
                                                    docs_selected,
                                                    initial_ranking,
                                                    topic_score_matrix,
                                                    topic_frequency)
        # Sim(d_i, q)
        # relevance = _lookup_rel(initial_ranking, doc_current)
        # print(rel)
        mmr_current = (lambda_score * first_score) + ((1 - lambda_score) * second_score)
        # print(mmr_current)
        # argmax mmr
        if mmr_current > mmr:
            mmr = mmr_current
            doc = d
        else:
            continue
    return mmr, doc


def rank(initial_ranking, lambda_score):
    """Ranking based on mmr score."""

    topic_distribution_np = np.array([json.loads(item) for item in initial_ranking['cluster_prob']])
    # topic_distribution = initial_ranking['cluster_prob']
    # topic_distribution_np = topic_distribution.to_numpy()
    final_ranking = [{
        'doc': initial_ranking['doc'].tolist()[0],
        'mmr': 'top',
        'cluster_scores': topic_distribution_np[0].tolist(),
        'cluster': topic_distribution_np.argmax(),
    }]

    print('--- topic frequency ---')
    print('-' * 100)
    for item in zip(*np.unique(topic_distribution_np.argmax(-1), return_counts=True)):
        print('topic: {} --> freq: {}'.format(item[0], item[1]))
    print('-' * 100)

    docs_unranked = initial_ranking['doc'].tolist()[1:]
    # topic_distribution = topic_distribution

    for curr_doc in tqdm(docs_unranked):
        mmr_score, doc = _mmr(
            lambda_score,
            curr_doc,
            docs_unranked,
            final_ranking,
            initial_ranking,
            topic_distribution_np
        )

        # cluster_scores = topic_distribution_np[initial_ranking.index[initial_ranking['doc'] == curr_doc]]
        # cluster = cluster_scores.argmax(-1).tolist()
        final_ranking.append(
            {'doc': doc,
             'mmr': mmr_score,
             # 'cluster': cluster[0],
             # 'cluster_scores': cluster_scores.tolist()[0],
             }
        )
        docs_unranked.remove(doc)
        docs_unranked.append(curr_doc)

    return final_ranking


def run_exp(initial_ranking, lambda_):
    ranked_documents = rank(initial_ranking, lambda_)
    chapter_rank = []
    for r_doc in ranked_documents:
        matched_doc_ = initial_ranking['doc'] == r_doc['doc']
        index_id = initial_ranking.index[matched_doc_].tolist()[0]

        chapter, text, rank_org, label, cluster_scores = initial_ranking.values.tolist()[index_id]

        chapter_rank.append([chapter, text, rank_org, label, r_doc['mmr'], cluster_scores])

    cols = ['chapter', 'text', 'score', 'label', 'mmr', 'cluster_scores']
    return pd.DataFrame(chapter_rank, columns=cols)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda_score', default=0.9, help='higher value of lambda means less topic diversity')
    parser.add_argument('--input_file', default='data/rank_v2_sim_0.1.csv')
    parser.add_argument('--output_file', default='data/rank_topic_0.9_full.csv')
    args = parser.parse_args()

    df = pd.read_csv(args.input_file)
    # df['cluster_prob'] = df['cluster_prob'].apply(json.loads)
    df = [item[1].reset_index(drop=True) for item in df.groupby(by='chapter')]

    list_of_df = [run_exp(item, args.lambda_score) for item in df[:2]]

    df = pd.concat(list_of_df)

    df.to_csv(args.output_file, index=False)
