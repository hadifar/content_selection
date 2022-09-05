from sentence_transformers import SentenceTransformer
import numpy as np

from clustering.autogmm import AutoGMMCluster

model = SentenceTransformer('all-MiniLM-L6-v2')


def _build_sim_matrix(initial_ranking):
    # Compute embeddings
    embeddings = model.encode([item for item in initial_ranking['doc'].tolist()], convert_to_tensor=True)
    # Compute cosine-similarities for each sentence with each other sentence
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings.data.numpy()
    # gm = GaussianMixture(n_components=n_cluster, n_init=10, reg_covar=0.05, random_state=42).fit(embeddings)
    gm = AutoGMMCluster(min_components=2,
                        max_components=10,
                        kmeans_n_init=10,
                        max_iter=1000,
                        # affinity='cosine',
                        # covariance_type='full',
                        # linkage='complete',
                        random_state=42,
                        n_jobs=-1).fit(embeddings)
    print(gm.n_components_, gm.affinity_, gm.covariance_type_, gm.linkage_, )
    # cluster_prob = gm.predict_proba(embeddings)
    # print(cluster_prob)


print()
print()
print()

import pandas as pd

df = pd.read_csv('../data/rank_v2.csv')
df = df.rename(columns={"text": "doc", "pred": 'score'})
df = [g[1] for g in df.groupby('chapter')]

i = 0
for df_item in df:
    df_item = df_item[df_item['score'] > 0.]
    print('chapter {}'.format(i + 1))
    _build_sim_matrix(df_item)
    print('-' * 100)
    i += 1
