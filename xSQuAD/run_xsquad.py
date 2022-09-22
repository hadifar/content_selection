import json

import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse


def topic_relevance(doc_current, docs_selected, topic_distribution_matrix, topic_frequency):
    n_examples, _ = topic_distribution_matrix.shape
    final_score = 0

    # curr_doc_index = initial_ranking['doc'] == doc_current
    curr_doc_rank_score = doc_current[2]  # original rank score
    curr_doc_topic_scores = doc_current[4]  # topic distribution score

    for tf in topic_frequency:
        topic, freq = tf
        topic_weight = (freq / n_examples)

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


def _mmr(lambda_score, doc_current, docs_unranked, docs_selected, topic_score_matrix):
    mmr = -1
    doc = None
    topic_frequency = np.asarray(np.unique(topic_score_matrix.argmax(-1), return_counts=True)).T

    for d in docs_unranked:
        # argmax Sim(d_i, d_j)
        pointwise_score, topicwise_score = topic_relevance(d,
                                                           docs_selected,
                                                           topic_score_matrix,
                                                           topic_frequency)

        mmr_current = (lambda_score * pointwise_score) + ((1 - lambda_score) * topicwise_score)

        if mmr_current > mmr:
            mmr = mmr_current
            doc = d
        else:
            continue

    return mmr, doc


def rank(initial_ranking, lambda_score):
    """Ranking based on mmr score."""

    topic_distribution_np = np.array(initial_ranking['cluster_prob'].tolist())

    final_ranking = [{
        'doc': initial_ranking.values[0].tolist(),
        'mmr': 'top',
    }]

    docs_unranked = initial_ranking.values[1:].tolist()

    for curr_doc in tqdm(docs_unranked):
        mmr_score, doc = _mmr(
            lambda_score=lambda_score,
            doc_current=curr_doc,
            docs_unranked=docs_unranked,
            docs_selected=final_ranking,
            topic_score_matrix=topic_distribution_np,
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


def run_exp(initial_ranking, lambda_, chapter_indx):
    print('chapter {}'.format(chapter_indx))
    # initial_ranking = initial_ranking.drop(columns=['covariance_type', 'linkage', 'n_components', 'affinity'])
    initial_ranking['score'] = initial_ranking['score'].rank() / len(initial_ranking)
    initial_ranking = initial_ranking.reset_index(drop=True)
    ranked_documents = rank(initial_ranking, lambda_)
    ranked_documents = [rdoc_['doc'] + [rdoc_['mmr']] for rdoc_ in ranked_documents]
    cols = ['chapter', 'text', 'score', 'label', 'cluster_scores', 'mmr']
    return pd.DataFrame(ranked_documents, columns=cols)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda_score', default=1.0, help='higher value of lambda means less topic diversity')
    parser.add_argument('--input_file', default='data/rank_v3_20_topics_len_threshold_25.csv')
    # parser.add_argument('--output_file', default='data/rank_topic_0.1_full.csv')
    args = parser.parse_args()
    for v in [0.01, 0.001, 0.0001, 0.05, 0.005, 0.0005, 0.09, 0.009, 0.0009]:
        args.lambda_score = v
        df = pd.read_csv(args.input_file)
        df['cluster_prob'] = df['cluster_prob'].apply(json.loads)
        df = [item[1] for item in df.groupby(by='chapter')]

        list_of_df = [run_exp(item, args.lambda_score, i) for i, item in enumerate(df)]

        df = pd.concat(list_of_df)

        df.to_csv('data/rank_topic_based_{}_len25.csv'.format(args.lambda_score), index=False)
