import numpy as np
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
#
# sns.set(style="whitegrid")
#
# rs = np.random.RandomState(365)
# values = rs.randn(365, 4).cumsum(axis=0)
# dates = pd.date_range("1 1 2016", periods=365, freq="D")
# data = pd.DataFrame(values, dates, columns=["A", "B", "C", "D"])
# data = data.rolling(7).mean()
#
# # 创建折线图
# sns.lineplot(data=data, palette="tab10", linewidth=2.5);
# plt.show()
# print(":::::");

import matplotlib.pyplot as plt

x1 = [iday for iday in range(0,1000)]
y1 = np.load('rmse202251333350.npy', allow_pickle=True)
y2 = np.load('groundtruth_7.npy', allow_pickle=True)
y3 = np.load('pred_0.npy', allow_pickle=True)
y4 = np.load('rmse_yzj2022513143637.npy', allow_pickle=True)

plt.plot(x1,y1,'r^-',label='RMSE', markersize=1);
plt.plot(x1,y4,'r^-',label='RMSE1', markersize=1);

plt.show()
print("1111");
# fig1 = plt.figure(1)
# axes = plt.subplot(111)
# axes.show()