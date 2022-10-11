import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mglearn
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from sklearn.datasets import make_blobs
from matplotlib import pyplot
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


data = pd.read_csv("C:/Users/11453/PycharmProjects/riskassessment/data/creditrisk/application_train.csv")
section=data[['AMT_GOODS_PRICE','AMT_ANNUITY']]
# 缺失值处理+编码
for col in section:
    if section[col].dtype == 'object':
        section[col].fillna(section[col].mode()[0], inplace=True)  # 使用众数填充标称型
    else:
        # df[col].fillna(round(df[col].mean()), inplace=True)
        section[col].fillna(0, inplace=True)  # 使用中位数填充数值型median


from sklearn.cluster import KMeans
k = 10 # 定义聚类的类别中心个数，即聚成4类
iteration = 500 # 计算聚类中心的最大循环次数
model = KMeans(n_clusters = k,max_iter = iteration)
model.fit(section)
text = model.predict(section)
data_frame = pd.DataFrame(text,index=None, columns=['test'])
"""r1 = pd.Series(model.labels_).value_counts()
r2 = pd.DataFrame(model.cluster_centers_)
r = pd.concat([r2,r1],axis=1)
r.columns = list(section.columns)+[u'所属类别数目']
print(r)"""

