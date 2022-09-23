import pandas as pd
import json
from itertools import chain

all_data = []
with open('../mTQA_test.json') as inpfile:
    dataset = json.load(inpfile)
    for i, chapter in enumerate(dataset):
        for q in chapter['questions']:
            all_data.append([i, q['question_text'], q['ground_sentence']])

    df = pd.DataFrame(all_data, columns=['chapter', 'question', 'context'])
    df.to_csv('qg_ground_question_context_pairs.csv', index=False)
