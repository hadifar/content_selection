import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import json


def load_data():
    # todo: at this moment we only return 0 chapter
    with open('../raw_data/qg_valid.json') as inpfile:
        ds = json.load(inpfile)
        return [" ".join([" ".join(s) for s in para]) for para in ds[2]['tokenized_chapter_text']]


embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Corpus with example sentences
corpus = load_data()
corpus_embeddings = embedder.encode(corpus)

# Normalize the embeddings to unit length
corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

gm = GaussianMixture(n_components=10, n_init=10, covariance_type='diag').fit(corpus_embeddings[1:])
pred = gm.predict(corpus_embeddings)

for item in zip(pred, corpus):
    print(item)
    print('-' * 50)

# print(pred)
# pred  = gm.predict_proba(corpus_embeddings[:1])
# print(pred)
