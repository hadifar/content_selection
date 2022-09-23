import json
from itertools import chain
import pandas as pd
import numpy as np

with open('qg_valid.json') as inpfile2:
    raw_ds = json.load(inpfile2)
    all_df = []
    for i, chapter in enumerate(raw_ds):
        list_of_q = [q['question']['question_text'] for q in chapter['questions']]
        df = pd.DataFrame()
        df['chapter'] = [i] * len(list_of_q)
        df['question'] = list_of_q
        df['context'] = [q['hl_context'].replace('<hl>','') for q in chapter['questions']]
        all_df.append(df)

all_df = pd.concat(all_df)
all_df.to_csv('qg_ground_question_context_pairs.csv', index=False)
