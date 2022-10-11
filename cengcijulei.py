from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

data = pd.read_csv("C:/Users/11453/PycharmProjects/riskassessment/data/creditrisk/application_train.csv")
section=data[['AMT_GOODS_PRICE','AMT_ANNUITY']]

# 缺失值处理+编码
for col in section:
    if section[col].dtype == 'object':
        section[col].fillna(section[col].mode()[0], inplace=True)  # 使用众数填充标称型
    else:
        # df[col].fillna(round(df[col].mean()), inplace=True)
        section[col].fillna(0, inplace=True)  # 使用中位数填充数值型median

Y = linkage(section.values, method='complete', metric='euclidean')
# 输入阈值获取聚类的结果（fcluster返回每个点所属的cluster的编号）
cluster_assignments = fcluster(Y, t=0.5, criterion='distance')

print('Cluster assignments:', cluster_assignments)

"""""
# np.where根据cluster编号取点的索引
clusters = [np.where(i == cluster_assignments)[0].tolist() for i in range(1, cluster_assignments.max() + 1)]

print('Clusters:', clusters)

# 绘制聚类结果
for indices in clusters:
    plt.scatter(data[indices][:, 0], data[indices][:, 1])
plt.show()
"""""