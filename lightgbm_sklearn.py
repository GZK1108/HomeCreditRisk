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

    # 查找类型为非数值型标签
    df.select_dtypes('object').apply(pd.Series.nunique, axis=0)

    # 缺失值处理+编码
    for col in df:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)  # 使用众数填充标称型
            df[[col]] = df[[col]].apply(LabelEncoder().fit_transform)
        else:
            # df[col].fillna(round(df[col].mean()), inplace=True)
            df[col].fillna(0, inplace=True)  # 使用中位数填充数值型median
    # print(df.isnull().sum().sort_values())

    # 对原有数据集特征进行运算，得到新特征
    df['cerdit_annuity_ratio'] = df.apply(lambda x: x['AMT_CREDIT'] / x['AMT_ANNUITY'], axis=1)
    df['prices_income_ratio'] = df.apply(lambda x: x['AMT_GOODS_PRICE'] / x['AMT_INCOME_TOTAL'], axis=1)
    df['employed_age_ratio'] = df.apply(lambda x: x['DAYS_EMPLOYED'] / x['DAYS_BIRTH'], axis=1)
    df['credit_goods_ratio'] = df.apply(lambda x: x['AMT_CREDIT'] / x['AMT_GOODS_PRICE'], axis=1)

    # 相关系数计算
    all_correlations = df.corr(method='pearson')
    # 绘制热力图
    """plt.figure(figsize=(16, 12), dpi=80)
    sns.heatmap(data=correlations, annot=False, center=0)
    plt.show()"""

    # 查找标签与TARGET相关性
    target_orrelations = (abs(all_correlations['TARGET']).sort_values(ascending=True))
    # print(target_orrelations)

    """
        # 按相关系数删除标签
        for i in target_orrelations.items():
        if i[1] <= 0.005:  # 删除与TARGET相关性低于0.005的标签
            df.drop([i[0]], axis=1, inplace=True)"""
    # 按个数删除标签
    count = 0
    for i in target_orrelations.items():
        count = count + 1
        if count < 0:  # 删除与TARGET相关性低的30个标签10
            df.drop(i[0], axis=1, inplace=True)

    # df.to_csv('C:/Users/11453/PycharmProjects/riskassessment/data/creditrisk/creditdata.csv', index=False)
    return df


def lightgbm(df):
    y = df.iloc[:, 0]
    x = df.iloc[:, 1:]
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
    # train_x = train_x.values

    # lgb_train = lgb.Dataset(x_train, y_train, feature_name=labels)
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
    gbm = lgb.LGBMClassifier(objective='binary',
                             boosting_type='goss',
                             metric='auc',
                             max_depth=6,
                             num_leaves=40,
                             learning_rate=0.00545,
                             subsample_for_bin=240000,
                             colsample_bytree=0.48888,
                             reg_alpha=0.44,
                             reg_lambda=0.48,
                             min_split_gain=0.024766,
                             subsample=1,
                             is_unbalance=False,
                             num_iterations=5000,
                             )

    """parameters = {
        'max_depth': [4, 6, 8],
        'num_leaves': [20, 30, 40],
    }

    gsearch = GridSearchCV(params, param_grid=parameters, scoring='roc_auc', cv=3)
    gsearch.fit(x_train, y_train)
    print('参数的最佳取值:{0}'.format(gsearch.best_params_))
    print('最佳模型得分:{0}'.format(gsearch.best_score_))
    print(gsearch.cv_results_['mean_test_score'])
    print(gsearch.cv_results_['params'])"""

    # 训练
    testlist = [(x_test,y_test)]
    model = gbm.fit(x_train, y_train, eval_set=testlist, early_stopping_rounds=4500)
    # model = gbm.fit(x_train, y_train)


    # 保存重要特征
    """importance = gbm.feature_importance(importance_type='gain')
    feature_name = gbm.feature_name()

    feature_importance = pd.DataFrame({
        'feature_name': feature_name, 'importance': importance})
    feature_importance.sort_values(by=['importance'], ascending=False, inplace=True)
    # print(feature_importance)
    # feature_importance.to_csv('feature_importance.csv', index=False)"""

    # 检验
    y_pred = model.predict(x_test)
    preds = model.predict_proba(x_test)[:, 1]
    fpr1, tpr1, thresholds1 = roc_curve(y_test, preds, pos_label=1)  # pos_label=1
    print("AUC为:", auc(fpr1, tpr1))

    plt.title("lightgbm")
    plt.plot(fpr1, tpr1, label='ROC')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    # plt.show()


if __name__ == '__main__':
    time_start = time.time()
    temp = character()
    lightgbm(temp)
    time_end = time.time()
    time_sum = time_end - time_start
    print("运行时间:")
    print(time_sum)
    # joblib.dump(model, "xgboostpredict.j1")