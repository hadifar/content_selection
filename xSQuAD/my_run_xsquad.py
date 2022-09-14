import json

import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse


def topic_relevance(doc_current, docs_selected, topic_distribution_matrix, topic_frequency):
    n_examples, total_n_topic = topic_distribution_matrix.shape
    final_score = 0

    # curr_doc_index = initial_ranking['doc'] == doc_current
    curr_doc_rank_score = doc_current[2]  # original rank score
    curr_doc_topic_scores = doc_current[4]  # topic distribution score

    for tf in topic_frequency:
        topic, freq = tf
        topic_weight = 1/total_n_topic
        # topic_weight = (freq / n_examples)

        topic_importance_for_curr_doc = topic_weight * curr_doc_rank_score * curr_doc_topic_scores[topic]

        product_score_for_selected = 1
        for s in docs_selected:
            # selected_doc_index = initial_ranking['doc'] == s['doc']
            selected_doc_rank_score = s['doc'][2]  # original rank score
            selected_doc_topic_scores = s['doc'][4]  # topic distribution score

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


# def _mmr(lambda_score, doc_current, docs_unranked, docs_selected, topic_score_matrix):


def rank(initial_ranking, lambda_score):
    """Ranking based on mmr score."""

    topic_distribution_np = np.array(initial_ranking['cluster_prob'].tolist())
    print('--- topic frequency ---')
    print('-' * 100)
    for item in zip(*np.unique(topic_distribution_np.argmax(-1), return_counts=True)):
        print('topic: {} --> freq: {}'.format(item[0], item[1]))
    print('-' * 100)

    final_ranking = [{
        'doc': initial_ranking.values[0].tolist(),
        'mmr': 'top',
        # 'cluster_scores': topic_distribution_np[0].tolist(),
        # 'cluster': topic_distribution_np.argmax(),
    }]

    docs_unranked = initial_ranking.values[1:].tolist()
    # topic_distribution = topic_distribution

    topic_frequency = np.asarray(np.unique(topic_distribution_np.argmax(-1), return_counts=True)).T
    while len(final_ranking) != len(initial_ranking):

        mmr = -1
        doc = None
        for d in docs_unranked:
            # argmax Sim(d_i, d_j)
            pairwise_score, topicwise_score = topic_relevance(d,
                                                              final_ranking,
                                                              topic_distribution_np,
                                                              topic_frequency)
            # Sim(d_i, q)
            # relevance = _lookup_rel(initial_ranking, doc_current)
            # print(rel)
            mmr_current = (lambda_score * pairwise_score) + ((1 - lambda_score) * topicwise_score)
            # print(mmr_current)
            # argmax mmr
            if mmr_current > mmr:
                mmr = mmr_current
                doc = d

        final_ranking.append(
            {'doc': doc,
             'mmr': mmr,
             # 'cluster': cluster[0],
             # 'cluster_scores': cluster_scores.tolist()[0],
             }
        )

        docs_unranked.remove(doc)

    return final_ranking


def run_exp(initial_ranking, lambda_):
    initial_ranking['score'] = initial_ranking['score'].rank() / len(initial_ranking)
    ranked_documents = rank(initial_ranking, lambda_)
    # chapter_rank = []
    # for r_doc in ranked_documents:
    #     matched_doc_ = initial_ranking['doc'] == r_doc['doc']
    #     index_id = initial_ranking.index[matched_doc_].tolist()[0]
    #
    #     chapter, text, rank_org, label, cluster_scores = initial_ranking.values.tolist()[index_id]
    #
    #     chapter_rank.append([chapter, text, rank_org, label, r_doc['mmr'], cluster_scores])
    ranked_documents = [rdoc_['doc'] + [rdoc_['mmr']] for rdoc_ in ranked_documents]
    cols = ['chapter', 'text', 'score', 'label', 'cluster_scores', 'mmr']
    return pd.DataFrame(ranked_documents, columns=cols)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda_score', default=1.0, help='higher value of lambda means less topic diversity')
    parser.add_argument('--input_file', default='data/rank_v2_sim_0.0_20_topics.csv')
    parser.add_argument('--output_file', default='data/rank_topic_1.0_full.csv')
    args = parser.parse_args()

    for i in [0, 0.001, 0.005, 0.1, 0.5, 1]:
        args.lambda_score = i
        args.output_file = 'data/rank_topic_{}_full.csv'.format(i)
        df = pd.read_csv(args.input_file)
        df['cluster_prob'] = df['cluster_prob'].apply(json.loads)
        df = [item[1].reset_index(drop=True) for item in df.groupby(by='chapter')]

        list_of_df = [run_exp(item, args.lambda_score) for item in df]

        df = pd.concat(list_of_df)

        df.to_csv(args.output_file, index=False)
