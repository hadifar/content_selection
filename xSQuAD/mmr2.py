import json

import numpy as np
from tqdm import tqdm


# model = SentenceTransformer('all-MiniLM-L6-v2')


# def _build_topic_distribution(initial_ranking):
#     # Compute embeddings
#     embeddings = model.encode([item for item in initial_ranking['doc'].tolist()], convert_to_tensor=True)
#     # Compute cosine-similarities for each sentence with each other sentence
#     embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
#     embeddings = embeddings.data.numpy()
#
#     # gm = GaussianMixture(n_components=2, n_init=10, reg_covar=0.05, random_state=42).fit(embeddings)
#     #
#     gm = AutoGMMCluster(min_components=2,
#                         max_components=10,
#                         kmeans_n_init=10,
#                         max_iter=1000,
#                         # affinity=None,
#                         # covariance_type='diag',
#                         # linkage=None,
#                         random_state=42,
#                         n_jobs=-1).fit(embeddings)
#
#     cluster_prob = gm.predict_proba(embeddings)
#     # clusters = cluster_prob.argmax(-1)
#     return pd.DataFrame(cluster_prob)
#

# def _lookup_rel(initial_ranking, doc):
#     """Lookup table for relevance."""
#     return initial_ranking.loc[initial_ranking['doc'] == doc, 'score'].iloc[0]
# def _lookup_sim(doc1, doc2, sim_matrix, initial_ranking):
#     """Lookup pairwise similarity."""
#
#     doc1_idx = initial_ranking.index[initial_ranking['doc'] == doc1].tolist()[0]
#     doc2_idx = initial_ranking.index[initial_ranking['doc'] == doc2].tolist()[0]
#     sim_doc1_doc2 = sim_matrix.iat[doc1_idx, doc2_idx]
#     return sim_doc1_doc2


def topic_relevance(doc_current, docs_selected, initial_ranking, topic_distribution_matrix):
    n_examples, n_topics = topic_distribution_matrix.shape
    topic_frequency = np.asarray(np.unique(topic_distribution_matrix.argmax(-1), return_counts=True)).T
    final_score = 0

    curr_doc_index = initial_ranking['doc'] == doc_current
    curr_doc_rank_score = initial_ranking[curr_doc_index]['score'].tolist()[0]
    curr_doc_topic_scores = topic_distribution_matrix[initial_ranking.index[curr_doc_index].tolist()[0]]
    for tf in topic_frequency:
        topic, freq = tf
        topic_weight = (freq / n_examples)
        # topic_score_per_doc = 1

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
    for d in docs_unranked:
        # argmax Sim(d_i, d_j)
        first_score, second_score = topic_relevance(doc_current, docs_selected, initial_ranking, topic_score_matrix)
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
        'cluster': topic_distribution_np[0].argmax(-1),
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

        cluster_scores = topic_distribution_np[initial_ranking.index[initial_ranking['doc'] == curr_doc]]
        cluster = cluster_scores.argmax(-1).tolist()
        final_ranking.append(
            {'doc': doc,
             'mmr': mmr_score,
             'cluster': cluster[0],
             'cluster_scores': cluster_scores.tolist()[0],
             }
        )
        docs_unranked.remove(doc)
        docs_unranked.append(curr_doc)

    return final_ranking
