# -*- coding: utf-8 -*-
'''
@author: 知乎 小白lan
'''

"""A榜线上分数0.489811"""

import pandas as pd
import catboost as cat
from sklearn.model_selection import GridSearchCV

#load data
train = pd.read_csv('../data/train.csv')
test  = pd.read_csv('../data/test.csv')
subs  = pd.read_csv('../data/submit.csv')

#build model
#final parameters
cat_grid = {'depth':[6]
            , 'bootstrap_type':['Bernoulli']
            , 'od_type':['Iter']
            , 'l2_leaf_reg':[16]
            , 'learning_rate': [0.1]
           }
#search and fit
catgrid = GridSearchCV(cat.CatBoostClassifier(), param_grid=cat_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose = 10)
catgrid.fit( train.drop(['label','id'], axis=1), train['label'] )
 
#predict output prob
cat_proba = catgrid.predict_proba( test.drop(['id'], axis=1) )
prob_cat_df = pd.DataFrame( cat_proba )
prob_cat_df.columns = ['lb0','lb1']

#export submission
subs['label'] = prob_cat_df['lb1']
subs.to_csv('../subs/cat_grid_f1.csv',index=None)