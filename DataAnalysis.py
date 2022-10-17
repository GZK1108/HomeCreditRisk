import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("C:/Users/11453/PycharmProjects/riskassessment/data/creditrisk/application_train.csv")
df = df.drop(['SK_ID_CURR'], axis=1)
# df_str = df
# df_num = df

# 区分标称和数值
for col in df:
    if df[col].dtype == 'object':
        # df_num = df_num.drop(col, axis=1)
        df[[col]] = df[[col]].apply(LabelEncoder().fit_transform)
    else:
        df[col].fillna(round(df[col].mean()), inplace=True)
        # df_str = df_str.drop(col, axis=1)

df2 = df[['TARGET','APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG',
          'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG',
          'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG', 'APARTMENTS_MODE', 'BASEMENTAREA_MODE',
          'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE',
          'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE',
          'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI',
          'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI',
          'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI',
          'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI', 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'TOTALAREA_MODE',
          'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE'
          ]]

df3 = df[['TARGET','EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4',
          'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9',
          'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14',
          'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19',
          'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'FLAG_MOBIL',
          'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL',
          'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
          'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
          'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', ]]

df4 = df[['TARGET','AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
          'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
          'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR', 'ORGANIZATION_TYPE', 'OBS_30_CNT_SOCIAL_CIRCLE',
          'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
          'DEF_60_CNT_SOCIAL_CIRCLE', ]]

df1 = df.drop(['APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG',
               'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG',
               'LIVINGAPARTMENTS_AVG',
               'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG', 'APARTMENTS_MODE', 'BASEMENTAREA_MODE',
               'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE',
               'ENTRANCES_MODE',
               'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE',
               'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI',
               'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI',
               'ENTRANCES_MEDI',
               'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI',
               'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI', 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE',
               'TOTALAREA_MODE', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3',
               'FLAG_DOCUMENT_4',
               'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9',
               'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14',
               'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19',
               'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21',
               'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'FLAG_MOBIL',
               'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL',
               'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
               'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
               'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'AMT_REQ_CREDIT_BUREAU_HOUR',
               'AMT_REQ_CREDIT_BUREAU_DAY',
               'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
               'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR', 'ORGANIZATION_TYPE',
               'OBS_30_CNT_SOCIAL_CIRCLE',
               'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
               'DEF_60_CNT_SOCIAL_CIRCLE', ], axis=1)

# 相关系数计算
df1_correlations = df1.corr(method='pearson')
df2_correlations = df2.corr(method='pearson')
df3_correlations = df3.corr(method='pearson')
df4_correlations = df4.corr(method='pearson')
print(df1.columns)
# 绘制热力图
sns.set_style('whitegrid')
"""plt.figure(figsize=(18, 12), dpi=80)
mask = np.zeros_like(df1_correlations, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(data=df1_correlations, annot=True, center=0, mask=mask, fmt='.2f', cmap='GnBu',linewidths=0.1)
plt.show()"""
plt.figure(figsize=(18, 12), dpi=80)
sns.heatmap(data=df2_correlations, annot=False, cmap='YlGnBu')
plt.show()
"""plt.figure(figsize=(18, 12), dpi=80)
sns.heatmap(data=df3_correlations, annot=False, center=0)
plt.show()
plt.figure(figsize=(18, 12), dpi=80)
mask = np.zeros_like(df4_correlations, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(data=df4_correlations, annot=True, center=0, mask=mask, fmt='.2f', cmap='GnBu',linewidths=0.1)
plt.show()"""
"""df_num.fillna(0, inplace=True)
df_num['cerdit_annuity_ratio'] = df.apply(lambda x: x['AMT_CREDIT'] / x['AMT_ANNUITY'], axis=1)
df_num['prices_income_ratio'] = df.apply(lambda x: x['AMT_GOODS_PRICE'] / x['AMT_INCOME_TOTAL'], axis=1)
df_num['employed_age_ratio'] = df.apply(lambda x: x['DAYS_EMPLOYED'] / x['DAYS_BIRTH'], axis=1)
df_num['credit_goods_ratio'] = df.apply(lambda x: x['AMT_CREDIT'] / x['AMT_GOODS_PRICE'], axis=1)"""

"""for i in range(len(df_num.columns)-1):
    i = i+1
    plt.figure(figsize=(16, 8), dpi=100)
    # x = range(1, len(df_num)+1)
    x = df_num.iloc[:, 0]
    y = df_num.iloc[:, i]
    # 统计每个类型个数(计算量特别大)
    # count = Counter(y)
    # count_lst = sorted(count.items(), key=lambda s: (-s[1]))

    plt.scatter(x=x, y=y, marker='.', s=5)
    plt.xlabel('NUMBER')
    plt.ylabel(df_num.columns[i])
    plt.savefig(f'C:/Users/11453/PycharmProjects/riskassessment/data/creditrisk/pictureTARGET/{df_num.columns[i]}', dpi=100)
    plt.show()"""

# 单独绘图
# plt.figure(figsize=(16, 8), dpi=100)
"""x = range(1, len(df_num)+1)
y = df_num['cerdit_annuity_ratio']
# plt.scatter(x=x, y=y, marker='.', s=5)
y.hist(figsize=(12,10), bins=20)
plt.ylabel('NUMBER')
plt.xlabel('cerdit_annuity_ratio')
plt.show()"""
