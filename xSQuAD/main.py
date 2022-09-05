import pandas as pd
import mmr2


def main_fn(initial_ranking):
    res = mmr2.rank(initial_ranking, lambda_score)
    # print(len(res))
    temp_list = []
    for item in res:
        indexxx = initial_ranking.index[initial_ranking['doc'] == item['doc']].tolist()[0]
        # _ = clustering
        chapter, text, rank_org, label, _ = initial_ranking.values.tolist()[indexxx]
        # item['mmr']
        # print('text: ', text[:50])
        # print('rank_org', rank_org)
        # print('label', label)
        # print('mmr score', item['mmr'])
        # print(, rank_org, label, item['mmr'])
        # print('-' * 100)
        temp_list.append([chapter, text, rank_org, label, item['mmr'], item['cluster'], item['cluster_scores']])
    # df =
    return pd.DataFrame(temp_list, columns=['chapter', 'text', 'score', 'label', 'mmr', 'cluster', 'cluster_scores'])


# higher lambda score means less diversity
lambda_score = 0.9
# similarity_threshold = 0.2
df = pd.read_csv('../data/rank_v2_sim_0.1.csv')
# initial_ranking = initial_ranking.rename(columns={'text': 'doc', 'pred': 'score'})
# initial_ranking = [item[1].sort_values(by='score', ascending=False) for item in initial_ranking.groupby(by='chapter')]
# initial_ranking = initial_ranking[initial_ranking['score'] > similarity_threshold]
# initial_ranking = initial_ranking.reset_index(drop=True)
# initial_ranking['score'] = initial_ranking['score'].rank(ascending=True, pct=True)
df = [item[1].reset_index(drop=True) for item in df.groupby(by='chapter')]
# df = [df[1]]

list_of_df = [main_fn(item) for item in df]
df = pd.concat(list_of_df)
df.to_csv('rank_topic_{}_full.csv'.format(str(lambda_score)), index=False)
