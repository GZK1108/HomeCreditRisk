from sklearn.model_selection import train_test_split
import pandas as pd
from matplotlib import pyplot as plt
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_curve, recall_score, precision_score, auc
import time
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("C:/Users/11453/PycharmProjects/riskassessment/data/creditrisk/application_train.csv")


# 特征工程
def character():
    global df
    # 删除ID列
    df = df.drop(['SK_ID_CURR'], axis=1)
    # 计算数据缺失比例，并将结果保存到csv中
    value_cave = ((df.isnull().sum()) / df.shape[0]).sort_values(ascending=False).map(lambda x: "{:.2%}".format(x))
    # value_cave.to_csv('ValueNum.csv')

    # 删除数据缺失比例高于10%的特征
    for key, value in value_cave.items():
        if value > '10%':  # 查找数据缺失比例高于10%的项
            # 删除指定列
            df = df.drop([key], axis=1)

    # 输出当前数据集标签类型
    print(df.dtypes.value_counts())

    # 查找类型为非数值型标签
    nonnumb = df.select_dtypes('object').apply(pd.Series.nunique, axis=0)

    # 转换为数值型
    le = LabelEncoder()
    le_count = 0
    for col in df:
        if df[col].dtype == 'object':
            if len(list(df[col].unique())) <= 2:
                le.fit(df[col])
                df[col] = le.transform(df[col])
                le_count += 1
    # print('%d columns were label encoded.' % le_count)
    df = pd.get_dummies(df)
    print('Training Features shape: ', df.shape)

    # 缺失值处理
    print(df.isnull().sum().sort_values())

    # df['NAME_TYPE_SUITE'].fillna(df.NAME_TYPE_SUITE.mode()[0], inplace=True)
    for col in df:
        if df[col].dtype == 'object':
            df[col].fillna("NAN", inplace=True)  # 后期可以考虑使用众数填充
        else:
            df[col].fillna(0, inplace=True)
    print(df.isnull().sum().sort_values())
    return df

def lightgbm(df):
    y = df["TARGET"]
    x = df.drop("TARGET", axis="columns")
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
    params = {
        'max_depth': 5,
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'12', 'auc', 'binary_logloss'},
        'num_leaves': 13,
        'learning_rate': 0.02,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'is_unbalance': True,
        'force_row_wise': True
    }

    gbm = lgb.train(params, lgb_train, num_boost_round=1736, valid_sets=lgb_eval)
    # gbm = lgb.train(params, lgb_train, num_boost_round=1000, early_stopping_rounds=100)

    Y_pred = gbm.predict(x_test)

    """print("准确度为：")
    print(accuracy_score(test_y, Y_pred, normalize=True, sample_weight=None))
    print("精确度为:")
    print(precision_score(test_y, Y_pred, average='binary'))  # 测试集精确率
    print("召回率为:")
    print(recall_score(test_y, Y_pred, average="binary"))"""

    fpr1, tpr1, thresholds1 = roc_curve(y_test, Y_pred, pos_label=1)  # pos_label=1
    print("AUC为:")
    print(auc(fpr1, tpr1))

    plt.title("lightgbm")
    plt.plot(fpr1, tpr1, label='ROC')
    plt.xlabel('FPR')
    plt.ylabel('TPR')

    plt.show()



# 运行完查看是否有标签
if __name__ == '__main__':
    time_start = time.time()
    df = character()
    # lightgbm(df)
    time_end = time.time()
    time_sum = time_end - time_start
    print("运行时间:")
    print(time_sum)

