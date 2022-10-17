import numpy as np
import pandas as pd
import warnings
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, recall_score, precision_score, auc
import time
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from imblearn.over_sampling import ADASYN
import joblib
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")


def character():
    df = pd.read_csv("C:/Users/11453/PycharmProjects/riskassessment/data/creditrisk/application_train.csv")
    # 删除ID列与FLAG_MOBIL列
    df = df.drop(['SK_ID_CURR', 'FLAG_MOBIL'], axis=1)
    # 计算数据缺失比例，并将结果保存到csv中
    value_cave = ((df.isnull().sum()) / df.shape[0]).sort_values(ascending=False).map(lambda x: "{:.2%}".format(x))
    # value_cave.to_csv('ValueNum.csv')
    # df.info()  # 数据集信息

    # 删除数据缺失比例高于10%的特征
    for key, value in value_cave.items():
        if value > '70%':  # 查找数据缺失比例高于70%的项，在该数据集里面没有
            # print(key)
            # 删除指定列
            df = df.drop([key], axis=1)

    # 输出当前数据集标签类型
    # print(df.dtypes.value_counts())

    # 统计类别个数
    print(df['TARGET'].value_counts())

    # 查找类型为非数值型标签
    df.select_dtypes('object').apply(pd.Series.nunique, axis=0)

    # 归一化

    # 缺失值处理+编码
    for col in df:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)  # 使用众数填充标称型
            df[[col]] = df[[col]].apply(LabelEncoder().fit_transform)
        else:
            df[col].fillna(round(df[col].mean()), inplace=True)
            # df[col].fillna(0, inplace=True)  # 使用中位数填充数值型median
    # print(df.isnull().sum().sort_values())

    # 对原有数据集特征进行运算，得到新特征
    df['cerdit_annuity_ratio'] = df.apply(lambda x: x['AMT_CREDIT'] / x['AMT_ANNUITY'], axis=1)
    df['prices_income_ratio'] = df.apply(lambda x: x['AMT_GOODS_PRICE'] / x['AMT_INCOME_TOTAL'], axis=1)
    df['employed_age_ratio'] = df.apply(lambda x: x['DAYS_EMPLOYED'] / x['DAYS_BIRTH'], axis=1)
    df['credit_goods_ratio'] = df.apply(lambda x: x['AMT_CREDIT'] / x['AMT_GOODS_PRICE'], axis=1)
    df['ext_source'] = df.apply(lambda x: x['EXT_SOURCE_3'] + x['EXT_SOURCE_2'] + x['EXT_SOURCE_1'], axis=1)

    section = df[["cerdit_annuity_ratio", "EXT_SOURCE_3", "EXT_SOURCE_2", "EXT_SOURCE_1", "ext_source"]]
    # 聚类读取
    model = joblib.load('kmeans.txt')
    clusion = model.predict(section)
    df['classfication'] = clusion

    # df.to_csv('C:/Users/11453/PycharmProjects/riskassessment/data/creditrisk/creditdata.csv', index=False)
    return df



def xgb(df):
    y = df.iloc[:, 0]
    x = df.iloc[:, 1:]
    train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=42, test_size=0.2)
    # train_x = train_x.values

    model = XGBClassifier(scale_pos_weight=11,  # ;在各类别样本十分不平衡时，把这个参数设定为一个正值，可以使算法更快收敛。默认1
                          learning_rate=0.0545,  # 学习效率
                          n_estimators=5000,
                          max_depth=6,  # 树深度
                          gamma=0.18,  # 节点分类所需的最小损失函数下降值
                          subsample=1,  # 控制每棵树的随机采样比例
                          colsample_bytree=0.4888,  # 特征随机采样比例
                          objective='binary:logistic', # 二分类的逻辑回归问题，返回预测的概率
                          eval_metric='auc',
                          reg_alpha=0.44,
                          reg_lambda=0.48,
                          seed=27)
    model.fit(train_x, train_y)
    # pickle.dump(model, open("xgb.pickle.dat", "wb"))
    # 载入模型并保存

    # loaded_model = pickle.load(open("xgb.pickle.dat", "rb"))

    Y_pred = model.predict(test_x)
    preds = model.predict_proba(test_x)[:, 1]

    print('y-pred', Y_pred)

    print("准确度为：")
    print(accuracy_score(test_y, Y_pred, normalize=True, sample_weight=None))
    print("精确度为:")
    print(precision_score(test_y, Y_pred, average='binary'))  # 测试集精确率
    print("召回率为:")
    print(recall_score(test_y, Y_pred, average="binary"))

    fpr1, tpr1, thresholds1 = roc_curve(test_y, preds, pos_label=1)  # pos_label=1
    print("AUC为:")
    print(auc(fpr1, tpr1))

    plt.title("Xgboost")
    plt.plot(fpr1, tpr1, label='ROC')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()


if __name__ == '__main__':
    time_start = time.time()
    temp = character()
    xgb(temp)
    time_end = time.time()
    time_sum = time_end - time_start
    print("运行时间:")
    print(time_sum)
    # joblib.dump(model, "xgboostpredict.j1")
