# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import make_interp_spline
#
# recall_at_k = [0.6, 0.65, 0.5, 0.7, 0.4, 0.2, 0.8, 0.38, 0.48, 0.8]
# ADC_at_k = [0.3, 0.25, 0.15, 0.27, 0.144, 0.29, 0.81, 0.48, 0.78, 0.66]
#
# X_Y_Spline = make_interp_spline(range(len(recall_at_k)), recall_at_k)
# X_ = np.linspace(min(range(len(recall_at_k))), max(range(len(recall_at_k))), 500)
# Y_ = X_Y_Spline(X_)
# plt.plot(X_, Y_, label="Recall@10", marker='v')
#
#
# X_Y_Spline = make_interp_spline(range(len(ADC_at_k)), ADC_at_k)
# X_ = np.linspace(min(range(len(recall_at_k))), max(range(len(recall_at_k))), 500)
# Y_ = X_Y_Spline(X_)
# plt.plot(range(len(ADC_at_k)), ADC_at_k, label="ADC@10", marker='<')
# # plt.plot(range(len(evaluated_recalls)), robert_recall, label="Recall", marker='>')
# # plt.plot(range(len(evaluated_recalls)), mrr_recall, label="TopicWise", marker='^')
# plt.legend()
# # plt.xticks(ticks=[0, 2.25, 4.5, 7.25, 9], labels=[str(i) for i in [0., 0.25, 0.5, 0.75, 1]])
# # plt.yticks(ticks=[0.,0.25, 0.5, 0.75, 1], labels=[str(i) for i in [0.,0.25, 0.5, 0.75, 1]])
#
# plt.xlabel('Lambda')
# # plt.ylabel('Recall')
#
# plt.show()


import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# Dataset
x = np.array([0, 0.1, .2, .3, .4, .5, .6, .7, .8, 0.9, 1])
y = np.array([22.97, 28.10, 28.30, 28.51, 28.30, 28.46, 28.46, 28.66, 28.46, 28.25, 28.46])/100
# y = y / y.max(axis=-1)

X_Y_Spline = make_interp_spline(x, y)

# Returns evenly spaced numbers
# over a specified interval.
X_ = np.linspace(x.min(), x.max(), 100)
Y_ = X_Y_Spline(X_)
plt.plot(X_, Y_, label='R@10')

x = np.array([0, 0.1, .2, .3, .4, .5, .6, .7, .8, 0.9, 1])
y = np.array([61.60, 59.59, 59.43, 59.44, 59.35, 59.23, 59.25, 59.19, 59.13, 59.04, 59.06])/100
# y = y / y.max(axis=-1)

X_Y_Spline = make_interp_spline(x, y)

# Returns evenly spaced numbers
# over a specified interval.
X_ = np.linspace(x.min(), x.max(), 100)
Y_ = X_Y_Spline(X_)
plt.plot(X_, Y_, label='ADC@10')

# Plotting the Graph
plt.xlabel("Lambda")
plt.ylabel("Score")
plt.legend()
plt.show()
