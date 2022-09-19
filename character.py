import pandas as pd

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
    # nonnumb = df.select_dtypes('object').apply(pd.Series.nunique, axis=0)

    # 使用one hot encode 转换为数值型
    df = pd.get_dummies(df)
    # print(df.dtypes.value_counts())

    # 缺失值处理
    # print(df.isnull().sum().sort_values())

    # df['NAME_TYPE_SUITE'].fillna(df.NAME_TYPE_SUITE.mode()[0], inplace=True)
    for col in df:
        if df[col].dtype == 'object':
            df[col].fillna("NaN", inplace=True)  # 后期可以考虑使用众数填充
        else:
            # df[col].fillna(round(df[col].mean()), inplace=True)
            df[col].fillna("NaN", inplace=True)
    # print(df.isnull().sum().sort_values())
    df.to_csv('C:/Users/11453/PycharmProjects/riskassessment/data/creditrisk/creditdata.csv', index=False)



# 运行完查看是否有标签
if __name__ == '__main__':
    character()

