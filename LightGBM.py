import pandas as pd
from matplotlib import pyplot as plt
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_curve, recall_score, precision_score, auc
import time

# import joblib

time_start = time.time()
df = pd.read_csv('C:/Users/11453/PycharmProjects/riskassessment/TapNet/无处理/unbal.csv', header=1)
df2 = pd.read_csv('C:/Users/11453/PycharmProjects/riskassessment/TapNet/无处理/testdata.csv', header=1)

train_y = df.iloc[:, 0]
train_x = df.iloc[:, 1:]

test_x = df2.iloc[:, 1:]
test_y = df2.iloc[:, 0]

train_x = train_x.values
test_x = test_x.values
train_y = train_y.values
test_y = test_y.values

lgb_train = lgb.Dataset(train_x, train_y)
lgb_eval = lgb.Dataset(test_x, test_y, reference=lgb_train)

params = {
    'max_depth': 32,
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'12', 'auc', 'binary_logloss'},
    'num_leaves': 40,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'is_unbalance': True
}

gbm = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=lgb_eval, early_stopping_rounds=500)
# gbm = lgb.train(params, lgb_train, num_boost_round=1000, early_stopping_rounds=100)

# pickle.dump(model, open("xgb.pickle.dat", "wb"))
# 载入模型并保存

# loaded_model = pickle.load(open("xgb.pickle.dat", "rb"))

Y_pred = gbm.predict(test_x)

"""print("准确度为：")
print(accuracy_score(test_y, Y_pred, normalize=True, sample_weight=None))
print("精确度为:")
print(precision_score(test_y, Y_pred, average='binary'))  # 测试集精确率
print("召回率为:")
print(recall_score(test_y, Y_pred, average="binary"))"""

fpr1, tpr1, thresholds1 = roc_curve(test_y, Y_pred, pos_label=1)  # pos_label=1
print("AUC为:")
print(auc(fpr1, tpr1))

plt.title("lightgbm")
plt.plot(fpr1, tpr1, label='ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
time_end = time.time()
plt.show()

time_sum = time_end - time_start
print("运行时间:")
print(time_sum)

# joblib.dump(model, "xgboostpredict.j1")
