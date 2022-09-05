"""
This is a simple application for sentence embeddings: clustering
Sentences are mapped to sentence embeddings and then agglomerative clustering with a threshold is applied.
"""
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import json


def create_connectivity_graph(sample_query):
    roles = [item[0] for item in sample_query]
    rl = len(roles)
    connectivity = np.zeros((rl, rl))
    for i, r1 in enumerate(roles):
        for j, r2 in enumerate(roles):
            if i != j and abs(i - j) == 1:
                connectivity[i][j] = 1
    return connectivity


def load_data():
    # todo: at this moment we only return 0 chapter
    with open('../raw_data/qg_valid.json') as inpfile:
        ds = json.load(inpfile)
        return [" ".join([" ".join(s) for s in para]) for para in ds[0]['tokenized_chapter_text']]

embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Corpus with example sentences
corpus = load_data()
corpus_embeddings = embedder.encode(corpus)

# Normalize the embeddings to unit length
corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

# Perform kmean clustering
# AgglomerativeClustering(affinity='cosine', linkage='single', connectivity=con, n_clusters=2)
clustering_model = AgglomerativeClustering(affinity='cosine',
                                           linkage='single',
                                           connectivity=create_connectivity_graph(corpus),
                                           compute_full_tree=True,
                                           compute_distances=True,
                                           distance_threshold=0.8,
                                           n_clusters=None)  # , affinity='cosine', linkage='average', distance_threshold=0.4)
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_

# from sklearn.neighbors import NearestCentroid
# y_predict = clustering_model.fit_predict(corpus_embeddings)
# clf = NearestCentroid()
# clf.fit(corpus_embeddings, y_predict)
# print(clf.centroids_)

clustered_sentences = {}
for sentence_id, cluster_id in enumerate(cluster_assignment):
    if cluster_id not in clustered_sentences:
        clustered_sentences[cluster_id] = []

    clustered_sentences[cluster_id].append(corpus[sentence_id])

np.save('../analyze/cluster.sentenc.npy', clustered_sentences)
for i, cluster in clustered_sentences.items():
    print("Cluster ", i + 1)
    print(cluster)
    print("")
