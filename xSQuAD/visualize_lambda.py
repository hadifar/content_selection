import matplotlib.pyplot as plt
import numpy as np


recall_at_k = [0.6, 0.65, 0.5, 0.7, 0.4, 0.2, 0.8, 0.38, 0.48, 0.8]
ADC_at_k = [0.3, 0.25, 0.15, 0.27, 0.144, 0.29, 0.81, 0.48, 0.78, 0.66]

plt.plot(range(len(recall_at_k)), recall_at_k, label="Recall", marker='v')
plt.plot(range(len(ADC_at_k)), ADC_at_k, label="ADC", marker='<')
# plt.plot(range(len(evaluated_recalls)), robert_recall, label="Recall", marker='>')
# plt.plot(range(len(evaluated_recalls)), mrr_recall, label="TopicWise", marker='^')
plt.legend()
plt.xticks(ticks=[0, 2.25, 4.5, 7.25, 9], labels=[str(i) for i in [0., 0.25, 0.5, 0.75, 1]])
plt.yticks(ticks=[0.,0.25, 0.5, 0.75, 1], labels=[str(i) for i in [0.,0.25, 0.5, 0.75, 1]])

plt.xlabel('Lambda')
# plt.ylabel('Recall')

plt.show()
