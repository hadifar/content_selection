import pandas as pd
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import random

df = pd.read_csv('data/rank_v2_sim_0.0_20_topics.csv')
df = [item[1] for item in df.groupby(by='chapter')]

data_for_topics_vis = []
for i, item in enumerate(df):
    print('chapter {}'.format(i))
    cs_variablity = []
    for row in item.values.tolist():
        if row[3] == 1:  # already labeled as ground-truth
            cluster_prop = np.array(json.loads(row[-1]))
            cs_variablity.append(cluster_prop.argmax())
            # print(cluster_prop.argmax())
    print('appearance of paragraphs {} -in cs- {}'.format(len(set(cs_variablity)), len(json.loads(item.values[0][-1]))))
    data_for_topics_vis.append([i + 1, len(json.loads(item.values[0][-1])), len(set(cs_variablity))])
    print('-' * 100)


random.seed(3)
data_for_topics_vis = random.sample(data_for_topics_vis, 24)

dataset = pd.DataFrame(data_for_topics_vis, columns=['chapter', 'topics', 'paragraphs'])

plotviews = dataset.topics.plot(kind='bar', figsize=(9, 6), width=.65,
                                color='#034E61',
                                edgecolor='gray',
                                grid=False,
                                clip_on=False)

plotvisitors = dataset.paragraphs.plot(kind='bar', figsize=(9, 6), width=.3,
                                       color='#E05759',
                                       hatch="..",
                                       # edgecolor='black',
                                       grid=False, clip_on=False)

# Create proxy artist to generate legend
# views = plt.Rectangle((0, 0), 1, 1, fc="#278DBC", edgecolor='none')
# visitors = plt.Rectangle((0, 0), 1, 1, fc='#000099', edgecolor='none')
plt.xlabel('Chapter')
plt.xticks(rotation=0, ticks=range(len(df)), labels=list(range(1, len(df) + 1)))
plt.yticks(ticks=[i for i in range(1,21) if i%2==0], labels=[str(i) for i in range(1,21) if i%2==0])

plt.ylabel('Number of topics')

plt.legend(labels=['Total topics', 'Covered topics'])
plt.tight_layout()
plt.xlim(-0.8, len(dataset) - 0.2)
plt.savefig('topic_para_distribution.pdf')
plt.show()
