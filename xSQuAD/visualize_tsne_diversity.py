import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import json
import matplotlib.colors

from xSQuAD.utils import categorical_cmap

random.seed(76)

tmp_df = [g[1] for g in pd.read_csv('data/rank_v2_sim_0.0_20_topics.csv').groupby(by='chapter')]
tmp_df = [item for item in tmp_df if 2 <= len(json.loads(item['cluster_prob'].values[0])) <= 5]
tmp_df = random.sample(tmp_df, 6)

embedder = SentenceTransformer('all-MiniLM-L6-v2')

c3 = categorical_cmap(10, 1, cmap="tab10")


def obtain_data_for_draw(axes, sub_df):
    corpus_embeddings = embedder.encode([str(item) for item in sub_df['doc'].tolist()])
    corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    tsne = TSNE(n_components=2, random_state=42)
    z = tsne.fit_transform(corpus_embeddings)
    clusters_ = [json.loads(v) for v in sub_df['cluster_prob'].values]

    for subz, suby, clust in zip(z, sub_df['label'].tolist(), clusters_):
        if suby == 0:
            c_ = c3.colors[np.array(clust).argmax()]
            m_ = '*'
            ms_ = 24
            # label = 'content'
        else:
            c_ = ['k']
            m_ = 'P'
            ms_ = 28

        axes.scatter(
            x=[subz[0]], y=[subz[1]],
            marker=m_, c=c_,
            s=ms_,

        )
        axes.tick_params(
            direction='inout',
            # axis='both',          # changes apply to the x-axis
            # which='both',      # both major and minor ticks are affected
            # left=False,
            # right=False,
            # bottom=False,  # ticks along the bottom edge are off
            # top=False,  # ticks along the top edge are off
            labelbottom=False,
            labeltop=False,
            labelleft=False,
            labelright=False,
        )
        # axes.set_xticks([])
        # axes.set_yticks([])
        # axes.set_xlim(0, 10)
        # axes.set_ylim(0, 20)


def draw_subplot(all_df):
    f, axes = plt.subplots(nrows=2, ncols=3)
    for i, sub_df in enumerate(all_df[:3]):
        obtain_data_for_draw(axes[0][i], sub_df)
        # if i ==0:
        # axes[0][i].legend(['Selected content', 'content'], loc="upper left",)

    for i, sub_df in enumerate(all_df[3:]):
        obtain_data_for_draw(axes[1][i], sub_df)

    legend_elements = [
        Line2D([0], [0], marker='P', color='w',
               markerfacecolor='k', markersize=8),

        (Line2D([0], [0], marker='*', color='w',
                markerfacecolor=c3.colors[1], markersize=10),
         Line2D([0], [0], marker='*', color='w',
                markerfacecolor=c3.colors[0], markersize=10)

         ),

    ]

    axes[0][0].legend(legend_elements,
                      ['Selected', 'unSelected'],
                      loc='upper left',
                      # title='Contents',
                      handler_map={tuple: HandlerTuple(ndivide=None)})


# for sub_df in all_df:
draw_subplot(tmp_df)
plt.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.savefig('tsne_distribution.pdf')
plt.show()
