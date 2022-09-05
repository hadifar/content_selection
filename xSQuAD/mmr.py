import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


def _build_sim_matrix(initial_ranking):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Compute embeddings
    embeddings = model.encode([item for item in initial_ranking['doc'].tolist()], convert_to_tensor=True)
    # Compute cosine-similarities for each sentence with each other sentence
    cosine_scores = util.cos_sim(embeddings, embeddings)
    return pd.DataFrame(cosine_scores.data.numpy())

def _lookup_rel(initial_ranking, doc):
    """Lookup table for relevance."""
    return initial_ranking.loc[initial_ranking['doc'] == doc, 'score'].iloc[0]


def _lookup_sim(doc1, doc2, sim_matrix, initial_ranking):
    """Lookup pairwise similarity."""
    try:
        doc1_idx = initial_ranking.index[initial_ranking['doc'] == doc1].tolist()[0]
        doc2_idx = initial_ranking.index[initial_ranking['doc'] == doc2].tolist()[0]
    except IndexError:
        print('index error')
        return 0
    sim_doc1_doc2 = sim_matrix.iat[doc1_idx, doc2_idx]
    return sim_doc1_doc2


def _mmr(lambda_score, doc_current, docs_unranked, docs_selected, initial_ranking, sim_matrix):
    """Compute mmr"""
    mmr = -10000
    doc = None
    for d in docs_unranked:
        # argmax Sim(d_i, d_j)
        sim = 0
        for s in docs_selected:
            sim_current = _lookup_sim(
                doc_current, s['doc'],
                sim_matrix,
                initial_ranking)
            if sim_current > sim:
                sim = sim_current
            else:
                continue
        # Sim(d_i, q)
        rel = _lookup_rel(initial_ranking, doc_current)
        # print(rel)
        mmr_current = lambda_score * rel - (1 - lambda_score) * sim
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
    final_ranking = [{'doc': initial_ranking['doc'].tolist()[0], 'mmr': 'top'}]

    sim_matrix = _build_sim_matrix(initial_ranking)

    docs_unranked = initial_ranking['doc'].tolist()[1:]
    for curr_doc in tqdm(docs_unranked):
        mmr_score, doc = _mmr(
            lambda_score,
            curr_doc,
            docs_unranked,
            final_ranking,
            initial_ranking,
            sim_matrix
        )

        final_ranking.append({'doc': doc, 'mmr': mmr_score})
        docs_unranked.remove(doc)
        docs_unranked.append(curr_doc)

    return final_ranking
