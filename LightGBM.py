import pandas as pd
import warnings
from lightgbm import plot_importance
from matplotlib import pyplot as plt
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_curve, recall_score, precision_score, auc
import time
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
# import joblib
warnings.filterwarnings("ignore")

time_start = time.time()
df = pd.read_csv('C:/Users/11453/PycharmProjects/riskassessment/data/creditrisk/creditdata.csv', header=0)

y = df.iloc[:, 0]
x = df.iloc[:, 1:]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
# train_x = train_x.values

# lgb_train = lgb.Dataset(x_train, y_train, feature_name=labels)
lgb_train = lgb.Dataset(x_train, y_train)
lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
params = {
    'boosting_type': 'goss',  # gbdt使用树，goss使用单边梯度抽样算法，使用随机森林
    'metric': 'auc',
    'use_missing': True,
    'learning_rate': 0.005134,
    'num_leaves': 54,
    'max_depth': 10,
    'subsample_for_bin': 240000,
    'reg_alpha': 0.436193,
    'reg_lambda': 0.479169,
    'colsample_bytree': 0.508716,
    'min_split_gain': 0.024766,
    'subsample': 1,
    'is_unbalance': True,
    'silent': -1,
    'verbose': -1
}

gbm = lgb.train(params, lgb_train, num_boost_round=3000, valid_sets=lgb_eval, early_stopping_rounds=2800)
# gbm = lgb.train(params, lgb_train, num_boost_round=1000, early_stopping_rounds=100)

# 输出特征重要性

# plot_importance(gbm, max_num_features=20, importance_type='gain')
# plt.show()


# 保存重要特征
importance = gbm.feature_importance(importance_type='gain')
feature_name = gbm.feature_name()

feature_importance = pd.DataFrame({
    'feature_name': feature_name, 'importance': importance})
feature_importance.sort_values(by=['importance'], ascending=1, inplace=True)
# print(feature_importance)
feature_importance.to_csv('feature_importance.csv', index=False)

# 删除低分特征并重新训练
"""for i in range(len(feature_importance)):
    if feature_importance['importance'][i] == 0:
        df.drop([feature_importance['feature_name'][i]], axis=1, inplace=True)

y = df.iloc[:, 0]
x = df.iloc[:, 1:]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
lgb_train = lgb.Dataset(x_train, y_train)
lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
gbm2 = lgb.train(params, lgb_train, num_boost_round=2000, valid_sets=lgb_eval, early_stopping_rounds=1500)"""



"""# 检验
y_pred = gbm.predict(x_test)


fpr1, tpr1, thresholds1 = roc_curve(y_test, y_pred, pos_label=1)  # pos_label=1
print("AUC为:")
print(auc(fpr1, tpr1))

plt.title("lightgbm")
plt.plot(fpr1, tpr1, label='ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')

plt.show()

time_end = time.time()
time_sum = time_end - time_start
print("运行时间:")
print(time_sum)"""

# joblib.dump(model, "xgboostpredict.j1")
