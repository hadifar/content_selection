import json
from nltk import sent_tokenize
from itertools import chain
import pandas as pd


def find_chapter_(s, all_sent):
    for sent in all_sent:
        if sent[1] == s:
            return sent[0]

    Exception('fuck', s)


with open('../../../analyze/results_tqa_paragraph_selection_9_23_12_55_ep3.0_lr5e-05_seed42_eval_scores.json') as inpfile, open(
        '../mTQA_test.json') as inpfile2:
    book = json.load(inpfile2)
    # print(raw_ds)
    rank_results = json.load(inpfile)
    rank_results = rank_results['valid_pred']

    all_data = []
    for i, b in enumerate(book):
        for p in b['chapter_text_list']:
            for s in sent_tokenize(p):
                all_data.append([i, s])

    new_data = []
    for r in rank_results:
        c = find_chapter_(r[0][0], all_data)
        # chapter, txt, label, pred
        # new_data.append([c, r[0][0], r[1][1], r[0][1]])
        new_data.append([c, r[0][0], r[1][0], r[0][1]])

    df = pd.DataFrame(new_data, columns=['chapter', 'text', 'pred', 'label'])
    df.to_csv('tqa_rank_v2.csv', index=False)

    # print(new_data)
