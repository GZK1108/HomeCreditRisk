import numpy as np
import pandas as pd
import warnings
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, recall_score, precision_score, auc
import time
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import sklearn.svm as svm
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from imblearn.over_sampling import ADASYN
import joblib

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


def anasyn(df):
    X = df.iloc[:, 1:]  # X为解释变量集
    y = df.iloc[:, 0]  # y为结果集
    print('Original dataset shape %s' % Counter(y))

    ada = ADASYN(sampling_strategy='auto', random_state=42)  # 42 Control the randomization of the algorithm.
    x_res, y_res = ada.fit_resample(X, y)
    print('Resampled dataset shape %s' % Counter(y_res))
    return x_res, y_res


def svmm(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
    # train_x = train_x.values

    model = svm.SVC(kernel="linear", decision_function_shape="ovo")  # 核函数可选"linear", "poly", "rbf", "sigmoid"
    model.fit(x_train, y_train)
    joblib.dump(model, "svm.txt")
    acu_train = model.score(x_train, y_train)
    acu_test = model.score(x_test, y_test)
    target_pred = model.predict(x_test)
    recall = recall_score(y_test, target_pred, average="macro")

    print(acu_train)  # 训练集正确率
    print("svm的正确率,精确率,召回率分别为:")
    print(acu_test)  # 测试集正确率
    print(precision_score(y_test, target_pred, average='macro'))  # 测试集精确率
    print(recall)  # 召回率

    # svm roc
    fpr, tpr, thresholds = roc_curve(y_test, model.decision_function(x_test))
    print("svm的auc为:")
    print(auc(fpr, tpr))
    # plt.subplot(1,2,1)
    plt.plot(fpr, tpr, label='ROC')
    plt.title("svm")
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()


if __name__ == '__main__':
    time_start = time.time()
    temp = character()
    x, y = anasyn(temp)
    svmm(x, y)
    time_end = time.time()
    time_sum = time_end - time_start
    print("运行时间:")
    print(time_sum)
    # joblib.dump(model, "xgboostpredict.j1")
