import numpy as np
from graspologic.cluster import AutoGMMCluster
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')


def _build_topic_distribution(initial_ranking):
    # Compute embeddings
    embeddings = model.encode([item for item in initial_ranking['doc'].tolist()],batch_size=1)
    # Compute cosine-similarities for each sentence with each other sentence
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings.data.numpy()

    # gm = GaussianMixture(n_components=2, n_init=10, reg_covar=0.05, random_state=42).fit(embeddings)
    #
    gm = AutoGMMCluster(min_components=2,
                        max_components=10,
                        kmeans_n_init=10,
                        max_iter=1000,
                        # affinity=None,
                        # covariance_type='diag',
                        # linkage=None,
                        random_state=42,
                        n_jobs=-1).fit(embeddings)

    cluster_prob = gm.predict_proba(embeddings)
    # clusters = cluster_prob.argmax(-1)
    return pd.DataFrame(cluster_prob)
