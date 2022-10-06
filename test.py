params = {
        'boosting_type': 'goss',  # gbdt使用树，goss使用单边梯度抽样算法，使用随机森林
        'metric': 'auc',
        'learning_rate': 0.005,
        'num_leaves': 54,
        'max_depth': 10,
        'subsample_for_bin': 240000,
        'reg_alpha': 0.436193,
        'reg_lambda': 0.479169,
        'colsample_bytree': 0.508716,
        'min_split_gain': 0.024766,
        'subsample': 1,
        'is_unbalance': False,
        'silent':-1,
        'verbose':-1
    }