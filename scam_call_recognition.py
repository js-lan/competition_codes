# -*- coding: utf-8 -*-
'''
@author: 知乎 小白lan
'''

import os
import gc
import time
import psutil
import datetime
import numpy as np
import pandas as pd
import catboost as cat
import lightgbm as lgb
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier
from scipy.stats import entropy, pearsonr, stats
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
 
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.max_columns', None)
pd.set_option("display.max_colwidth",100)
pd.set_option('display.max_rows', None)
pd.set_option('display.width',100)
 
path = "./0527/"
feat_path = path + "data/"
 
def get_app_feats(df):
    phones_app = df[["phone_no_m"]].copy()
    phones_app = phones_app.drop_duplicates(subset=['phone_no_m'], keep='last')
    tmp = df.groupby("phone_no_m")["busi_name"].agg(busi_count="nunique")
    phones_app = phones_app.merge(tmp, on="phone_no_m", how="left")
    """使用的流量统计
    """
    tmp = df.groupby("phone_no_m")["flow"].agg(flow_mean="mean", 
                                               flow_median = "median", 
                                               flow_min  = "min", 
                                               flow_max = "max", 
                                               flow_var = "var",
                                               flow_skew = "skew",
                                               flow_std = "std",
                                               flow_quantile = "quantile",
                                               flow_sem = "sem",
                                               flow_sum = "sum")
    phones_app = phones_app.merge(tmp, on="phone_no_m", how="left")
    tmp = df.groupby("phone_no_m")["month_id"].agg(month_ids ="nunique")
    phones_app = phones_app.merge(tmp, on="phone_no_m", how="left")
    #月流量使用统计
    phones_app["flow_month"] = phones_app["flow_sum"] / phones_app["month_ids"]
    return phones_app
 
def get_voc_feat(df):
    df["start_datetime"] = pd.to_datetime(df['start_datetime'] )
    df["hour"] = df['start_datetime'].dt.hour
    df["day"] = df['start_datetime'].dt.day
    phone_no_m = df[["phone_no_m"]].copy()
    phone_no_m = phone_no_m.drop_duplicates(subset=['phone_no_m'], keep='last')
    #对话人数和对话次数
    tmp = df.groupby("phone_no_m")["opposite_no_m"].agg(opposite_count="count", opposite_unique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    """主叫通话
    """
    df_call = df[df["calltype_id"]==1].copy()
    tmp = df_call.groupby("phone_no_m")["imei_m"].agg(voccalltype1="count", imeis="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    phone_no_m["voc_calltype1"] = phone_no_m["voccalltype1"] / phone_no_m["opposite_count"] 
    tmp = df_call.groupby("phone_no_m")["city_name"].agg(city_name_call="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    tmp = df_call.groupby("phone_no_m")["county_name"].agg(county_name_call="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    
    """和固定通话者的对话统计
    """
    tmp = df.groupby(["phone_no_m","opposite_no_m"])["call_dur"].agg(count="count", sum="sum")
    phone2opposite = tmp.groupby("phone_no_m")["count"].agg(phone2opposite_mean="mean"
                                                            , phone2opposite_median="median"
                                                            , phone2opposite_max="max"
                                                            , phone2opposite_min="min"
                                                            , phone2opposite_var="var"
                                                            , phone2opposite_skew="skew"
                                                            , phone2opposite_sem="sem"
                                                            , phone2opposite_std="std"
                                                            , phone2opposite_quantile="quantile"
                                                           )
    phone_no_m = phone_no_m.merge(phone2opposite, on="phone_no_m", how="left")
    phone2opposite = tmp.groupby("phone_no_m")["sum"].agg(phone2oppo_sum_mean="mean"
                                                          , phone2oppo_sum_median="median"
                                                          , phone2oppo_sum_max="max"
                                                          , phone2oppo_sum_min="min"
                                                          , phone2oppo_sum_var="var"
                                                          , phone2oppo_sum_skew="skew"
                                                          , phone2oppo_sum_sem="sem"
                                                          , phone2oppo_sum_std="std"
                                                          , phone2oppo_sum_quantile="quantile"
                                                         )
    phone_no_m = phone_no_m.merge(phone2opposite, on="phone_no_m", how="left")
    
    """通话时间长短统计
    """
    tmp = df.groupby("phone_no_m")["call_dur"].agg(call_dur_mean="mean"
                                                   , call_dur_median="median"
                                                   , call_dur_max="max"
                                                   , call_dur_min="min"
                                                   , call_dur_var="var"
                                                   , call_dur_skew="skew"
                                                   , call_dur_sem="sem"
                                                   , call_dur_std="std"
                                                   , call_dur_quantile="quantile"
                                                  )
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    
    tmp = df.groupby("phone_no_m")["city_name"].agg(city_name_nunique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    tmp = df.groupby("phone_no_m")["county_name"].agg(county_name_nunique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    tmp = df.groupby("phone_no_m")["calltype_id"].agg(calltype_id_unique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    
    """通话时间点偏好
    """
    tmp = df.groupby("phone_no_m")["hour"].agg(voc_hour_mode = lambda x:stats.mode(x)[0][0], 
                                               voc_hour_mode_count = lambda x:stats.mode(x)[1][0], 
                                               voc_hour_nunique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    
    tmp = df.groupby("phone_no_m")["day"].agg(voc_day_mode = lambda x:stats.mode(x)[0][0], 
                                               voc_day_mode_count = lambda x:stats.mode(x)[1][0], 
                                               voc_day_nunique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    return phone_no_m
 
def get_sms_feats(df):
    df['request_datetime'] = pd.to_datetime(df['request_datetime'] )
    df["hour"] = df['request_datetime'].dt.hour
    df["day"] = df['request_datetime'].dt.day
 
    phone_no_m = df[["phone_no_m"]].copy()
    phone_no_m = phone_no_m.drop_duplicates(subset=['phone_no_m'], keep='last')
    #对话人数和对话次数
    tmp = df.groupby("phone_no_m")["opposite_no_m"].agg(sms_count="count", sms_nunique="nunique")
    tmp["sms_rate"] = tmp["sms_count"]/tmp["sms_nunique"]
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    """短信下行比例
    """
    calltype2 = df[df["calltype_id"]==2].copy()
    calltype2 = calltype2.groupby("phone_no_m")["calltype_id"].agg(calltype_2="count")
    phone_no_m = phone_no_m.merge(calltype2, on="phone_no_m", how="left")
    phone_no_m["calltype_rate"] = phone_no_m["calltype_2"] / phone_no_m["sms_count"]
    """短信时间
    """
    tmp = df.groupby("phone_no_m")["hour"].agg(hour_mode = lambda x:stats.mode(x)[0][0], 
                                               hour_mode_count = lambda x:stats.mode(x)[1][0], 
                                               hour_nunique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    
    tmp = df.groupby("phone_no_m")["day"].agg(day_mode = lambda x:stats.mode(x)[0][0], 
                                               day_mode_count = lambda x:stats.mode(x)[1][0], 
                                               day_nunique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    
    return phone_no_m
 
def feats():
    test_voc=pd.read_csv(path+'test/test_voc.csv',)
    test_voc_feat = get_voc_feat(test_voc)
    test_voc_feat.to_csv(feat_path + "test_voc_feat.csv", index=False)
 
    test_app=pd.read_csv(path+'test/test_app.csv',)
    test_app_feat = get_app_feats(test_app)
    test_app_feat.to_csv(feat_path + "test_app_feat.csv", index=False)
    
    test_sms=pd.read_csv(path+'test/test_sms.csv',)
    test_sms_feat = get_sms_feats(test_sms)
    test_sms_feat.to_csv(feat_path + "test_sms_feat.csv", index=False)
     
    train_voc=pd.read_csv(path+'train/train_voc.csv',)
    train_voc_feat = get_voc_feat(train_voc)
    train_voc_feat.to_csv(feat_path + "train_voc_feat.csv", index=False)
 
    train_app=pd.read_csv(path+'train/train_app.csv',)
    train_app_feat = get_app_feats(train_app)
    train_app_feat.to_csv(feat_path + "train_app_feat.csv", index=False)
 
    train_sms=pd.read_csv(path+'train/train_sms.csv',)
    train_sms_feat = get_sms_feats(train_sms)
    train_sms_feat.to_csv(feat_path + "train_sms_feat.csv", index=False)
    
    
    test_vocfs=pd.read_csv(path + 'zpfsdata/test_voc.csv',)
    test_voc_featfs = get_voc_feat(test_vocfs)
    test_voc_featfs.to_csv(path + "zpfsdata/test_voc_feat.csv", index=False)
 
    test_appfs=pd.read_csv(path + 'zpfsdata/test_app.csv',)
    test_app_featfs = get_app_feats(test_appfs)
    test_app_featfs.to_csv(path + "zpfsdata/test_app_feat.csv", index=False)
    
    test_smsfs=pd.read_csv(path + 'zpfsdata/test_sms.csv',)
    test_sms_featfs = get_sms_feats(test_smsfs)
    test_sms_featfs.to_csv(path + "zpfsdata/test_sms_feat.csv", index=False)
  
#create and save voc、app、sms features
feats()

#load april features
test_app_feat=pd.read_csv(feat_path+'test_app_feat.csv')
test_voc_feat=pd.read_csv(feat_path+'test_voc_feat.csv')
test_sms_feat=pd.read_csv(feat_path + "test_sms_feat.csv")
test_user=pd.read_csv(path+'test/test_user.csv')
test_user = test_user.merge(test_app_feat, on="phone_no_m", how="left")
test_user = test_user.merge(test_voc_feat, on="phone_no_m", how="left")
test_user = test_user.merge(test_sms_feat, on="phone_no_m", how="left")
test_user["city_name"] = LabelEncoder().fit_transform(test_user["city_name"].astype(np.str))
test_user["county_name"] = LabelEncoder().fit_transform(test_user["county_name"].astype(np.str))
#load april label
test_user_lb1 = pd.read_csv(path + 'zpfsdata/4yuelabel1.csv')
test_user_lb2 = pd.read_csv(path + 'zpfsdata/4yuelabel2.csv')
#concat april label and merge with features 
test_user_label = pd.concat([test_user_lb1, test_user_lb2])
test_user = test_user.merge(test_user_label, on="phone_no_m", how="left")
test_user.rename(columns={"arpu_202004":"arpu_202005"},inplace=True)
 
#load train features and label
train_app_feat = pd.read_csv(feat_path + "train_app_feat.csv")
train_voc_feat = pd.read_csv(feat_path + "train_voc_feat.csv")
train_sms_feat = pd.read_csv(feat_path + "train_sms_feat.csv")
train_user=pd.read_csv(path+'train/train_user.csv')
drop_r = ["arpu_201908","arpu_201909","arpu_201910","arpu_201911","arpu_201912","arpu_202001","arpu_202002"]
train_user.drop(drop_r, axis=1,inplace=True)
train_user.rename(columns={"arpu_202003":"arpu_202005"},inplace=True)
train_user = train_user.merge(train_app_feat, on="phone_no_m", how="left")
train_user = train_user.merge(train_voc_feat, on="phone_no_m", how="left")
train_user = train_user.merge(train_sms_feat, on="phone_no_m", how="left")
train_user["city_name"] = LabelEncoder().fit_transform(train_user["city_name"].astype(np.str))
train_user["county_name"] = LabelEncoder().fit_transform(train_user["county_name"].astype(np.str))
 
#concat preli data(train and test)
train_user = pd.concat([train_user, test_user])
#final label
train_label = train_user[["label"]].copy()
 
#drop phone_no_m
test_user.drop(["phone_no_m"], axis=1,inplace=True)
train_user.drop(["phone_no_m", "label"], axis=1,inplace=True)
 
#load final test features as testfs, fs means fusai
test_app_featfs=pd.read_csv(path + 'zpfsdata/test_app_feat.csv')
test_voc_featfs=pd.read_csv(path + 'zpfsdata/test_voc_feat.csv')
test_sms_featfs=pd.read_csv(path + 'zpfsdata/test_sms_feat.csv')
test_userfs=pd.read_csv(path + 'zpfsdata/test_user.csv')
test_userfs = test_userfs.merge(test_app_featfs, on="phone_no_m", how="left")
test_userfs = test_userfs.merge(test_voc_featfs, on="phone_no_m", how="left")
test_userfs = test_userfs.merge(test_sms_featfs, on="phone_no_m", how="left")
test_userfs["city_name"] = LabelEncoder().fit_transform(test_userfs["city_name"].astype(np.str))
test_userfs["county_name"] = LabelEncoder().fit_transform(test_userfs["county_name"].astype(np.str))
 
#create submission dataframe
sub = test_userfs[["phone_no_m"]].copy()
#drop phone_no_m
test_userfs.drop(["phone_no_m"], axis=1,inplace=True)
test_userfs.replace([u'\\N'], np.nan, inplace=True)
test_userfs['arpu_202005'] = test_userfs['arpu_202005'].astype(np.float32)
test_userfs_ori = test_userfs

"""bulid cat lgb xgb model"""
depth = 8
cv = 5

#create catboost model
catclf = cat.CatBoostClassifier(
    allow_writing_files = False
    , od_type= 'Iter'
    , silent=True
)
#final parameters
cat_grid = {'depth':[depth]
            , 'bootstrap_type':['Bernoulli']
            , 'od_type':['Iter']
            , 'l2_leaf_reg':[15]
            , 'learning_rate': [0.1]
            , 'allow_writing_files':[False]
            , 'silent':[True]
 
           }
#search and fit
catgrid = GridSearchCV(cat.CatBoostClassifier(), param_grid=cat_grid, cv=cv, scoring='f1_macro', n_jobs=-1, verbose = 10)
catgrid.fit( train_user, train_label['label'] )
 
#predict output prob
test_userfs = test_userfs_ori.fillna( test_userfs_ori.quantile(0.39) )
test_userfs['arpu_202005'] = test_userfs['arpu_202005'].astype(np.float32)
cat_proba = catgrid.predict_proba( test_userfs )
rslt_prob_cat = pd.DataFrame( cat_proba )
rslt_prob_cat.columns = ['lb0','lb1']
 
#create lgb model
#final parameters
lgb_grid = {'booster':['gbdt']
            , 'num_leaves':[256]
            , 'min_child_weight':[4]
            , 'feature_fraction':[0.7]
            , 'bagging_fraction':[0.8]
            , 'bagging_freq': [1]
           }
 
#search and fit
lgbgrid = GridSearchCV(lgb.LGBMClassifier(), param_grid=lgb_grid, cv=cv, scoring='f1_macro', n_jobs=-1, verbose = 10)
lgbgrid.fit( train_user, train_label['label'] )
#predict output prob
test_userfs = test_userfs_ori.fillna( test_userfs_ori.quantile(0.34) )
test_userfs['arpu_202005'] = test_userfs['arpu_202005'].astype(np.float32)
lgb_proba = lgbgrid.predict_proba( test_userfs )
rslt_prob_lgb = pd.DataFrame(lgb_proba)
rslt_prob_lgb.columns = ['lb0','lb1']
 
#create xgb model
#final parameters
from xgboost import XGBClassifier
xgbclf=XGBClassifier(base_score=0.5
                     , booster='gbtree'
                     , colsample_bytree=0.9
                     , learning_rate=0.1
                     , max_depth=8
                     , min_child_weight=7
                     , n_estimators=100
                     , n_jobs=-1
                     , objective='binary:logistic'
                     , subsample=0.75
                     , verbosity=1)
#fit
xgbclf.fit( train_user, train_label['label'] )
#predict output prob
test_userfs = test_userfs_ori.fillna( test_userfs_ori.quantile(0.319) )
test_userfs['arpu_202005'] = test_userfs['arpu_202005'].astype(np.float32)
xgb_proba = xgbclf.predict_proba( test_userfs )
rslt_prob_xgb = pd.DataFrame(lgb_proba)
rslt_prob_xgb.columns = ['lb0','lb1']


"""模型融合"""
"""调整概率输出"""
bestnew112 =  0.25*rslt_prob_lgb + 0.25*rslt_prob_xgb + 0.5*rslt_prob_cat
 
bestnew112["label"]=bestnew112["lb1"]
bestnew112["label"][bestnew112.label>60/100]=1
bestnew112["label"][bestnew112.label<60/100]=0
    
sub['label'] = bestnew112['label']
 
print(sub['label'].value_counts())
print(sub['label'].value_counts()/sub.shape[0])
 
sub.to_csv('lgb25xgb25cat50threshold60.csv',index=None)