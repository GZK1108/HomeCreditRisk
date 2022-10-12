import numpy as np
import pandas as pd
import warnings
import seaborn as sns
from lightgbm import plot_importance
from matplotlib import pyplot as plt
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_curve, recall_score, precision_score, auc
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans, DBSCAN

df = pd.read_csv("C:/Users/11453/PycharmProjects/riskassessment/data/creditrisk/application_train.csv")
# 删除ID列与FLAG_MOBIL列
df = df.drop(['SK_ID_CURR', 'FLAG_MOBIL'], axis=1)

# 缺失值处理+编码
for col in df:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)  # 使用众数填充标称型
        df[[col]] = df[[col]].apply(LabelEncoder().fit_transform)
    else:
        # df[col].fillna(round(df[col].mean()), inplace=True)
        df[col].fillna(0, inplace=True)  # 使用中位数填充数值型median

section=df[['AMT_GOODS_PRICE','AMT_ANNUITY']]
k = 10 # 定义聚类的类别中心个数，即聚成4类
iteration = 500 # 计算聚类中心的最大循环次数
model = DBSCAN(eps=0.5, min_samples=5, metric_params=None, leaf_size=30, n_jobs=-1)
model.fit(section)
text = model.fit_predict(section)
df['test'] = text

print(df)