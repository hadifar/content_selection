import argparse
from random import sample
import torch
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm

from data_helper import read_json_file

model = T5ForConditionalGeneration.from_pretrained('hadifar/openstax_qg_agno')
tokenizer = T5Tokenizer.from_pretrained('hadifar/openstax_qg_agno')


def main(arg):
    dataset = read_json_file(arg.inp_file)
    # dataset = dataset[:1]

    all_df = []
    for i, chapter in tqdm(enumerate(dataset)):
        tmp = [[q['question']['question_text'], 1] for q in chapter['questions']]  # label 1 for ground_truth
        df = pd.DataFrame()
        paragraphs = chapter['chapter_text'].split('\n\n')
        paragraphs = sample(paragraphs, len(tmp) * 2)
        for ctx in paragraphs:
            encoding = tokenizer(ctx, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(**encoding)
            tokenized_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            tmp.append([tokenized_output, 0])  # label zero for generated
        df['chapter'] = len(tmp) * [i]
        df['questions'] = [t[0] for t in tmp]
        df['label'] = [t[1] for t in tmp]
        all_df.append(df)
    all_df = pd.concat(all_df)
    all_df.to_csv(args.out_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp_file', default='raw_data/qg_train.json')
    parser.add_argument('--out_file', default='raw_data/overgenerate_train.csv')
    # parser.add_argument('--inp_file', default=0.05, help='higher value of lambda means less topic diversity')
    # parser.add_argument('--output_file', default='data/rank_topic_0.1_full.csv')
    args = parser.parse_args()
    main(args)
