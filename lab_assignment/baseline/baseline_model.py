# Description: Baseline code for the House Price Prediction Competition
# File: model building

# import necessary files

import baseline_code 
import xgboost as xgb
import numpy as np
import pandas as pd

from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

from mlxtend.regressor import StackingCVRegressor



# loading test and test data
train, test = baseline_code.get_train_test_data()
y_label = baseline_code.get_train_label()

kfolds = KFold(n_splits = 10, shuffle = True, random_state = 61)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model, X = train):
    rmse = np.sqrt(-cross_val_score(model, X, y_label, scoring="neg_mean_squared_error", cv=kfolds))
    return (rmse)

alp2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008] 
# using linear regression
lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=alp2, random_state=42, cv=kfolds))

# using support vector machine algorithm
svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003,))

# regrassor
xgboost = xgb.XGBRegressor(learning_rate=0.01,n_estimators=3460,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006)

# suing gradient boosting algorithm
gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, 
max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =42) 

# using ensemble
stack_gen = StackingCVRegressor(regressors=(lasso, gbr, svr), 
                        meta_regressor=xgboost,
                        use_features_in_secondary=True)

# training our methods
score = cv_rmse(lasso)
print("LASSO: {:.4f} ({:.4f})\n".format(score.mean(), score.std()),  )

score = cv_rmse(svr)
print("SVR: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), )

score = cv_rmse(gbr)
print("gbr: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), )


# fitting ensemble of algorithms
print('stack_gen')
stack_gen_model = stack_gen.fit(np.array(train), np.array(y_label))

# predict
ensemble_predict = stack_gen_model.predict(np.array(test))
ensemble_predict = np.expm1(ensemble_predict)

# my submission
sub = pd.DataFrame()
sub['Id'] = baseline_code.get_test_ID()
sub['SalePrice'] = ensemble_predict
sub.to_csv('submission.csv',index=False)