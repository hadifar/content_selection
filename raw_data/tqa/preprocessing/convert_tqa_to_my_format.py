import json

import nltk.tokenize
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize

df = pd.read_csv('tqa-a.csv')
# counter = 0
# list_of_all = []
import re


def preprocess(chapter_list_contxt):
    new_chapter_list = []
    for c in chapter_list_contxt:
        valid_parantetis = [item for item in re.findall(r'\([^()]*\)', c) if
                            item.find('figure') != -1 or item.find('Figure') != -1]
        if len(valid_parantetis) != 0:
            for v in valid_parantetis:
                c = c.replace(v, '')
        if c and c != '' and " ".join(c.split()) != ' ':
            new_chapter_list.append(c)
    return new_chapter_list


def jaccard_sim(s1, s2):
    s1 = set(s1.split())
    s2 = set(s2.split())
    return len(s1.intersection(s2)) / len(s1.union(s2))


import numpy as np


def matched_sentece(c, txt_to_fin):
    all_sents = nltk.tokenize.sent_tokenize(c)
    match = [s for s in all_sents if s.find(txt_to_fin) != -1]
    if len(match) != 1:
        send_id = np.array([jaccard_sim(s, txt_to_fin) for s in all_sents]).argmax(-1)
        return all_sents[send_id]

    else:
        return match[0]


def find_sentence_in_Text(row, chapter_list_contxt):
    """
    dont touch this function, every block of code handle a specific case for the alignment
    (figure)
    Space.
    Space .
    """

    span_txt = row[1]['CorrectAnswerSpan']
    beg_ans = row[1]['CorrectAnswerStart']
    candidate_ = [s for s in chapter_list_contxt if s.find(span_txt) != -1]
    if len(candidate_) == 1:
        matched_sen = matched_sentece(candidate_[0], span_txt)
        return candidate_, span_txt, matched_sen

    end_of_beg_ans = row[1]['Text'][beg_ans:].find('.') + 1
    txt_to_find = row[1]['Text'][beg_ans:beg_ans + end_of_beg_ans]
    txt_to_find = " ".join(txt_to_find.split())
    candidate_ = [s for s in chapter_list_contxt if s.find(txt_to_find) != -1]
    if len(candidate_) == 1:
        matched_sen = matched_sentece(candidate_[0], span_txt)
        return candidate_, span_txt, matched_sen

    txt_to_find = " ".join(nltk.tokenize.word_tokenize(row[1]['Text'][beg_ans:beg_ans + end_of_beg_ans]))
    candidate_ = [s for s in chapter_list_contxt if s.find(txt_to_find) != -1]
    if len(candidate_) == 1:
        matched_sen = matched_sentece(candidate_[0], span_txt)
        return candidate_, span_txt, matched_sen

    txt_to_find = row[1]['Text'][row[1]['Text'][:beg_ans].rfind('.') + 1:beg_ans]
    txt_to_find = " ".join(txt_to_find.split())
    if txt_to_find != '':
        candidate_ = [s for s in chapter_list_contxt if s.find(txt_to_find) != -1]
        if len(candidate_) == 1:
            matched_sen = matched_sentece(candidate_[0], span_txt)
            return candidate_, span_txt, matched_sen

    txt_to_find = span_txt[:len(span_txt) // 2]
    if len(txt_to_find) > 30:
        candidate_ = [s for s in chapter_list_contxt if s.find(txt_to_find) != -1]
        if len(candidate_) == 1:
            matched_sen = matched_sentece(candidate_[0], span_txt)
            return candidate_, span_txt, matched_sen

    # beg__ = row[1]['Text'][row[1]['Text'][:beg_ans].rfind('.') + 1:beg_ans]
    txt_to_find = " ".join(nltk.word_tokenize(row[1]['Text'][beg_ans:beg_ans + end_of_beg_ans]))
    candidate_ = [s for s in chapter_list_contxt if " ".join(nltk.word_tokenize(s)).find(txt_to_find) != -1]
    if len(candidate_) == 1:
        matched_sen = matched_sentece(candidate_[0], span_txt)
        return candidate_, span_txt, matched_sen

    txt_to_find = row[1]['Text'][row[1]['Text'][:beg_ans].rfind('.') + 1:beg_ans]
    txt_to_find = " ".join(txt_to_find.split())[len(txt_to_find) // 2:]
    txt_to_find = txt_to_find + ' ' + span_txt
    candidate_ = [s for s in chapter_list_contxt if s.find(txt_to_find) != -1]
    if len(candidate_) == 1:
        matched_sen = matched_sentece(candidate_[0], span_txt)
        return candidate_, span_txt, matched_sen

    txt_to_find = row[1]['Text'][row[1]['Text'][:beg_ans].rfind('.') + 1:beg_ans]
    txt_to_find = " ".join(txt_to_find.split())[:len(txt_to_find) // 2]
    candidate_ = [s for s in chapter_list_contxt if s.find(txt_to_find) != -1]
    if len(candidate_) == 1:
        matched_sen = matched_sentece(candidate_[0], span_txt)
        return candidate_, span_txt, matched_sen

    end_of_beg_ans = row[1]['Text'][beg_ans:].find('.') + 1
    txt_to_find = row[1]['Text'][beg_ans:beg_ans + end_of_beg_ans]
    txt_to_find = " ".join(txt_to_find.split())
    txt_to_find = txt_to_find[: len(txt_to_find) // 2]
    candidate_ = [s for s in chapter_list_contxt if s.find(txt_to_find) != -1]
    if len(candidate_) == 1:
        matched_sen = matched_sentece(candidate_[0], span_txt)
        return candidate_, span_txt, matched_sen

    print('fuck')
    return candidate_, span_txt, []


# def find_in_paragraphs(s, alist_of_paragraph):
#     find_paragraph = [item for item in alist_of_paragraph if item.find(s) != -1]
#     return find_paragraph

for target in ['v2_test', 'v1_val', 'v1_train']:
    print('process file -->', target)
    list_of_all = []
    with open('tqa_{}.json'.format(target)) as inpfile:
        dataset = json.load(inpfile)
        # print(dataset)
        for d in dataset:
            # sub_df = df[df['lessonName'] == d['lessonName']]
            chapter = []
            sub_df = df[df["LessonId"] == d['globalID']]
            if len(sub_df) != 0:
                chapter_text_list = [c.get('content').get('text') for c in d['topics'].values() if c.get(
                    'content')]
                                    #todo: the logic will not work if you uncomment this line becuse of error in preprocessing

                                    #+ [c.get('content').get('text') for c in d['adjunctTopics'].values() if
                                  # c.get('content')]
                chapter_text_list = preprocess(chapter_text_list)
                for row in sub_df.iterrows():
                    ground_paragraph, span_txt, matched_ = find_sentence_in_Text(row, chapter_text_list)

                    paragraph_id = [i for i, item in enumerate(chapter_text_list) if item == ground_paragraph[0]][0]
                    question_txt = row[1]['QuestionText']

                    q_obj = {
                        'question_text': question_txt,
                        'ground_paragraph': paragraph_id,
                        'ground_sentence': matched_,
                        'choices': row[1]['AnswerCandidate'],
                        'answer_span': row[1]['CorrectAnswerSpan'],
                    }

                    chapter.append(q_obj)
                if len(chapter) != 0:
                    c_obj = {
                        'questions': chapter,
                        'chapter_id': d['globalID'],
                        'lesson_name': d['lessonName'],
                        'chapter_text_list': chapter_text_list
                    }
                    list_of_all.append(c_obj)

    # print(list_of_all)
    with open('../mTQA_{}.json'.format(target).replace('v1_', '').replace('v2_', ''), 'w') as outfile:
        json.dump(list_of_all, outfile)

print('merge files ...')
with open('../mTQA_train.json') as inpfile, open('../mTQA_val.json') as inpfile2, open('../mTQA_train_valid.json',
                                                                                       'w') as outfile:
    l1, l2 = json.load(inpfile), json.load(inpfile2)
    l3 = l1 + l2
    json.dump(l3, outfile)
