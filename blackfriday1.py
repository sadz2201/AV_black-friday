import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.cross_validation import train_test_split
import operator
from matplotlib import pylab as plt

#def create_feature_map(features):
#    outfile = open('xgb.fmap', 'w')
#    i = 0
#    for feat in features:
#        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
#        i = i + 1
#
#    outfile.close()
#    
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submission = pd.read_csv("sample_submission.csv")
submission.User_ID=test.User_ID
submission.Product_ID=test.Product_ID

train=train.fillna(0)
test=test.fillna(0)

Y=train.Purchase
train.drop('Purchase', axis=1, inplace=True)

#create_feature_map(train.columns.values)

train = np.array(train)
test = np.array(test)

for i in range(train.shape[1]):
    lbl = LabelEncoder()
    lbl.fit(list(train[:,i]) + list(test[:,i]) )
    train[:,i] = lbl.transform(train[:,i])
    test[:,i] = lbl.transform(test[:,i])
    
train = train.astype(float)
test = test.astype(float)

Xtrain,Xcv,Ytrain,Ycv=train_test_split(train,Y,test_size=0.33,random_state=619)

num_rounds = 10000
xgtest = xgb.DMatrix(test)
xgtrain = xgb.DMatrix(Xtrain,label=Ytrain)
xgcv=xgb.DMatrix(Xcv,label=Ycv)
xgtrain_full=xgb.DMatrix(train,label=Y)
watchlist = [(xgtrain, 'train'),(xgcv, 'val')]

params = {}
params["objective"] = "reg:linear"
params["eta"] = 0.05
params["subsample"] = 0.75
params["colsample_bytree"] = 0.75
params["seed"] = 619
params["silent"] = 0
params["max_depth"] = 10
params["eval_metric"] = "rmse"
#params["gamma"] = 1.75
#params["min_child_weight"] = 2.4
#params["scale_pos_weight"] = 5      
plst = list(params.items())

model1 = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=100)
xgb1= xgb.train(plst,xgtrain_full,num_boost_round=model1.best_iteration)
preds1=xgb1.predict(xgtest)

params = {}
params["objective"] = "reg:linear"
params["eta"] = 0.05
params["subsample"] = 1
params["colsample_bytree"] = 0.8
params["seed"] = 619
params["silent"] = 0
params["max_depth"] = 8
params["eval_metric"] = "rmse"
#params["gamma"] = 1.75
#params["min_child_weight"] = 2.4
#params["scale_pos_weight"] = 5      
plst = list(params.items())

model2 = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=100)
xgb2= xgb.train(plst,xgtrain_full,num_boost_round=model2.best_iteration)
preds2=xgb2.predict(xgtest)

params = {}
params["objective"] = "reg:linear"
params["eta"] = 0.05
params["subsample"] = 0.9
params["colsample_bytree"] = 0.75
params["seed"] = 619
params["silent"] = 0
params["max_depth"] = 10
params["eval_metric"] = "rmse"
#params["gamma"] = 1.75
#params["min_child_weight"] = 2.4
#params["scale_pos_weight"] = 5      
plst = list(params.items())

model3 = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=100)
xgb3= xgb.train(plst,xgtrain_full,num_boost_round=model3.best_iteration)
preds3=xgb3.predict(xgtest)

##importance = xgb1.get_fscore(fmap='xgb.fmap')
##importance = sorted(importance.items(), key=operator.itemgetter(1))
##df = pd.DataFrame(importance, columns=['feature', 'fscore'])
##df['fscore'] = df['fscore'] / df['fscore'].sum()
##
##plt.figure()
##df.plot()
##df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 10))
##plt.title('XGBoost Feature Importance')
##plt.xlabel('relative importance')
##plt.gcf().savefig('feature_importance_xgb.png')
#
submission.Purchase=(preds1+preds2+preds3)/3
neg_idx=submission[submission['Purchase']<500].index.tolist()
submission.loc[neg_idx,'Purchase']=500
submission=submission.set_index('User_ID')
submission.to_csv('sadz11.csv')