import matplotlib.pyplot as plt

evaluated_recalls = [1, 3, 5, 10, 15]
evaluated_adc = [2, 4, 6, 8, 10]
yaxis = [0.0, 0.25, 0.5, 0.75, 1]

randm_recall = [0.0, 0.0, 0.0, 0.1, 0.1]
lex_recall = [0.0, 0.0, 0.2, 0.2, 0.3]
robert_recall = [0.0, 0.333, 0.4, 0.7142, 0.7142]
mrr_recall = [0.0, 0.66, 0.4, 0.7142, 0.8571]

random_adc = [0.3660658597946167, 0.6106657385826111, 0.5947957436243693, 0.6142063140869141, 0.6704919815063477]
lex_adc = [0.10715878009796143, 0.19927331805229187, 0.2289909521738688, 0.23679950833320618, 0.2520060777664185]
robert_adc = [0.16943058371543884, 0.4331997334957123, 0.4581868251164754, 0.5360115766525269, 0.5224740982055665]
mrr_adc = [0.22961652278900146, 0.42646342515945435, 0.4938184420267741, 0.5632709860801697, 0.5614033699035644]

plt.figure(figsize=(8, 4))
plt.subplot(121)

plt.plot(range(len(evaluated_recalls)), randm_recall, marker='v')
plt.plot(range(len(evaluated_recalls)), lex_recall, marker='<')
plt.plot(range(len(evaluated_recalls)), robert_recall, marker='>')
plt.plot(range(len(evaluated_recalls)), mrr_recall, marker='^')

plt.xticks(ticks=range(len(evaluated_recalls)), labels=[str(i) for i in evaluated_recalls])
plt.yticks(ticks=yaxis, labels=[str(i) for i in yaxis])

plt.xlabel('K')
plt.ylabel('Recall')

plt.subplot(122)
plt.plot(range(len(evaluated_recalls)), random_adc, label="Random", marker='v')
plt.plot(range(len(evaluated_recalls)), lex_adc, label="LexRank", marker='<')
plt.plot(range(len(evaluated_recalls)), robert_adc, label="PairWise", marker='>')
plt.plot(range(len(evaluated_recalls)), mrr_adc, label="TopicWise", marker='^')
# plt.legend()
plt.xticks(ticks=range(len(evaluated_adc)), labels=[str(i) for i in evaluated_adc])
plt.yticks(ticks=yaxis, labels=[str(i) for i in yaxis])
plt.xlabel('K')
plt.ylabel('ADC')

plt.figlegend(loc='upper left', ncol=1, labelspacing=0.1, bbox_to_anchor=[0.1, 0.95])
plt.tight_layout(w_pad=1)
plt.savefig('diff_analysis.pdf')
plt.show()
