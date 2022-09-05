import numpy as np
import pandas as pd
from graspologic.cluster import AutoGMMCluster

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')


def _build_topic_distribution(chapter):
    # Compute embeddings
    paragraph_in_chapter = [item for item in chapter['doc'].tolist()]
    embeddings = model.encode(paragraph_in_chapter, convert_to_tensor=True)
    # Compute cosine-similarities for each sentence with each other sentence
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings.data.numpy()

    # gm = GaussianMixture(n_components=2, n_init=10, reg_covar=0.05, random_state=42).fit(embeddings)
    #
    gm = AutoGMMCluster(min_components=2,
                        max_components=10,
                        kmeans_n_init=10,
                        max_iter=1000,
                        affinity='euclidean',
                        covariance_type='diag',
                        linkage='ward',
                        random_state=42,
                        n_jobs=-1).fit(embeddings)

    cluster_prob = gm.predict_proba(embeddings)
    # clusters = cluster_prob.argmax(-1)
    chapter['cluster_prob'] = cluster_prob.tolist()
    return chapter


similarity_threshold = 0.2
initial_ranking = pd.read_csv('../data/rank_v2.csv')
initial_ranking = initial_ranking.rename(columns={'text': 'doc', 'pred': 'score'})
initial_ranking = [item[1].sort_values(by='score', ascending=False) for item in initial_ranking.groupby(by='chapter')]
initial_ranking = [item[item['score'] > similarity_threshold].reset_index(drop=True) for item in initial_ranking][:2]
res = [_build_topic_distribution(item) for item in initial_ranking]
df = pd.concat(res)
df.to_csv('rank_v2_sim_{}.csv'.format(similarity_threshold), index=False)
