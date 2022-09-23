import json
from itertools import chain

import numpy as np


with open('../../analyze/results_9_14_23_19_ep5.0_lr3e-05_seed42_eval_scores.json') as inpfile, open(
        'qg_valid.json') as inpfile2:
    raw_ds = json.load(inpfile2)
    pargraph_chapters = [[" ".join([" ".join(s) for s in para]) for para in item['tokenized_chapter_text']] for item in
                         raw_ds]
    # pargraph_chapters = chain(*pargraph_chapters)
    # print(raw_ds)
    valid = json.load(inpfile)['valid_pred']
    # valid = [[re.sub('[%s]' % re.escape(string.punctuation), ' ', item[0]), item[1]] for item in valid]

    valid_score_for_a_chapter = []
    for cind, chapter in enumerate(pargraph_chapters):
        tmpo = []
        for p in chapter:
            # p = re.sub('[%s]' % re.escape(string.punctuation), ' ', p)

            score = search_in_valid(valid, p, cind)
            tmpo.append(score)
        valid_score_for_a_chapter.append(tmpo)
        # break

    # ranked = sorted(valid_score_for_a_chapter, key=lambda x: x[1][0], reverse=True)
    ranked = [sorted(item, key=lambda x: x[1][0], reverse=True) for item in valid_score_for_a_chapter]
    np.save('../../analyze/rank.sentence.npy', ranked)
    # print(valid_score_for_a_chapter)
    # for item in
    # print(ds)
