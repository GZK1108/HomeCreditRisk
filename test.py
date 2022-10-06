import numpy as np
import pandas as pd
from lightgbm import plot_importance
from matplotlib import pyplot as plt
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_curve, recall_score, precision_score, auc
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('C:/Users/11453/PycharmProjects/riskassessment/data/creditrisk/application_train.csv', header=0)
class_le = LabelEncoder()
# nonumb = df.select_dtypes('object').apply(pd.Series.nunique, axis=0)
for col in df:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)  # 后期可以考虑使用众数填充
        df[[col]]=df[[col]].apply(LabelEncoder().fit_transform)

    else:
        df[col].fillna(round(df[col].mean()), inplace=True)
# df.to_csv('temp_creditdata.csv', index=False)
