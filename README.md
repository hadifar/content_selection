# Diverse Content Selection for Educational Question Generation

This repository contains the code for the presented paper in XXXX 2022:

```
@inproceedings{hadifar-etal-2023-diverse,
    title = "Diverse Content Selection for Educational Question Generation",
    author = "Hadifar, Amir  and
      Bitew, Semere Kiros  and
      Deleu, Johannes  and
      Hoste, Veronique  and
      Develder, Chris  and
      Demeester, Thomas",
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics: Student Research Workshop",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.eacl-srw.13",
    pages = "123--133",
    abstract = "Question Generation (QG) systems have shown promising results in reducing the time and effort required to create questions for students. Typically, a first step in QG is to select the content to design a question for. In an educational setting, it is crucial that the resulting questions cover the most relevant/important pieces of knowledge the student should have acquired. Yet, current QG systems either consider just a single sentence or paragraph (thus do not include a selection step), or do not consider this educational viewpoint of content selection. Aiming to fill this research gap with a solution for educational document level QG, we thus propose to select contents for QG based on relevance and topic diversity. We demonstrate the effectiveness of our proposed content selection strategy for QG on 2 educational datasets. In our performance assessment, we also highlight limitations of existing QG evaluation metrics in light of the content selection problem.",
}

```


[Diverse Content Selection for Educational Question Generation](https://aclanthology.org/2023.eacl-srw.13/)



## Install dependencies

`pip install -r requirement.txt`

## Paragraph selection

`sh run_exp_para.sh`

## Question ranking

`sh run_exp_ques.sh`

## Diversify ranker
