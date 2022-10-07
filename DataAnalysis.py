import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("C:/Users/11453/PycharmProjects/riskassessment/data/creditrisk/application_train.csv")
df = df.drop(['SK_ID_CURR', 'FLAG_MOBIL'], axis=1)
df_str = df
df_num = df

# 区分标称和数值
for col in df:
    if df[col].dtype == 'object':
        df_num = df_num.drop(col, axis=1)
    else:
        # df[col].fillna(round(df[col].mean()), inplace=True)
        df_str = df_str.drop(col, axis=1)

for i in range(len(df_num.columns)):
    i = i+1
    plt.figure(figsize=(16, 8),dpi=100)
    x = range(1, len(df_num)+1)
    # x = df_num.iloc[:, 0]
    y = df_num.iloc[:, i]
    plt.scatter(x=x, y=y, marker='.')
    plt.xlabel('NUMBER')
    plt.ylabel(df_num.columns[i])
    plt.show()
