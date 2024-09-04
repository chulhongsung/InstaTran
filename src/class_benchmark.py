#%%

import sys
import os

import pandas as pd
import numpy as np

df_train_total = pd.read_csv("../data/df_train_total.csv")
df_test_total = pd.read_csv("../data/df_test_total.csv")
df_merged = pd.read_csv("../data/df_merged.csv")

#%%
def generate_eval_arima(label_df, input_seq_len=48, tau=12):
    col_labels =  'wl_1018680' # ['wl_1018662', 'wl_1018680', 'wl_1018683', 'wl_1019630']
    
    tmp_df = np.array(label_df.loc[label_df['year'] == 2021, col_labels])
    
    n = tmp_df.shape[0] - input_seq_len - tau 
    
    conti_input = np.zeros((n, input_seq_len), dtype=np.float32)
    label = np.zeros((n, tau))

    for j in range(n):
        conti_input[j, :] = tmp_df[j:(j+input_seq_len)]
        label[j, :] = tmp_df[(j+input_seq_len):(j+input_seq_len+tau)]

    return conti_input, label
#%%
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.arima import ARIMA, AutoARIMA
from sktime.forecasting.theta import ThetaForecaster
# y = load_airline()
fh = ForecastingHorizon(np.arange(1, 13), is_relative=True)

arima_forecaster = ARIMA()
# arima_forecaster = AutoARIMA(sp=1, d=0, max_p=2, max_q=2, suppress_warnings=True) 
y_train = df_merged["wl_1018680"]

arima_forecaster.fit(y_train)

y_pred = arima_forecaster.predict_quantiles(fh=fh, alpha=[0.1, 0.5, 0.7, 0.9])
#%%
y_input, label = generate_eval_arima(df_merged)
#%%
arima_forecaster.update(y_input[2650])
arima_forecaster.predict_quantiles(fh=fh, alpha=[0.1, 0.5, 0.7, 0.9])
y_input[2650]
label[2650]
#%%
ets_forecaster = ExponentialSmoothing(
    trend='add', seasonal='multiplicative', sp=24
)  
ets_forecaster = AutoETS(auto=True, n_jobs=-1, sp=24)  
ets_forecaster.fit(y_train)
y_pred = ets_forecaster.predict_quantiles(fh=fh, alpha=[0.1, 0.5, 0.7, 0.9])

ets_forecaster.update(y_input[2650])
ets_forecaster.predict_quantiles(fh=fh, alpha=[0.1, 0.5, 0.7, 0.9])
#%%
theta_forecaster = ThetaForecaster(sp=24)  
theta_forecaster.fit(y_train)
y_pred = theta_forecaster.predict_quantiles(fh=fh, alpha=[0.1, 0.5, 0.7, 0.9])

theta_forecaster.update(y_input[2650])
theta_forecaster.predict_quantiles(fh=fh, alpha=[0.1, 0.5, 0.7, 0.9])
#%%
y_input.shape
arima_results = []

y_train = df_merged["wl_1018680"]
arima_forecaster.fit(y_train)

for i in range(y_input.shape[0]):
    arima_forecaster.update(y_input[i], update_params=False)
    tmp_result = arima_forecaster.predict_quantiles(fh=fh, alpha=[0.1, 0.5, 0.7, 0.9])
    
    arima_results.append(tmp_result.values[np.newaxis, ...])
    print(i)
    # len(arima_results)
# %%
arima_results = np.concatenate(arima_results, axis=0)

np.maximum(0.9 * (label - arima_results[..., 3]), (1-0.9)*(arima_results[..., 3] -label )).mean()/1000 # 0.0031
np.maximum(0.7 * (label - arima_results[..., 2]), (1-0.7)*(arima_results[..., 2] -label )).mean()/1000 # 0.0051
np.maximum(0.5 * (label - arima_results[..., 1]), (1-0.5)*(arima_results[..., 1] -label )).mean()/1000  # 0.0059

np.mean(label < arima_results[..., 3]), 0.9 - np.mean(label < arima_results[..., 3])  ### 0.638
np.mean(label < arima_results[..., 2]), 0.7 - np.mean(label < arima_results[..., 2]) ### 0.640
np.mean(label < arima_results[..., 1]), 0.5- np.mean(label < arima_results[..., 1]) ### 0.894

#%%
ets_results = []
ets_forecaster = AutoETS(auto=True, n_jobs=-1)  

ets_forecaster.fit(y_train)

for i in range(y_input.shape[0]):
    ets_forecaster.update(y_input[i], update_params=False)
    tmp_result = ets_forecaster.predict_quantiles(fh=fh, alpha=[0.1, 0.5, 0.7, 0.9])
    
    ets_results.append(tmp_result.values[np.newaxis, ...])
    print(i)
#%%
ets_results = np.concatenate(ets_results, axis=0)

np.maximum(0.9 * (label - ets_results[..., 3]), (1-0.9)*(ets_results[..., 3] -label )).mean()/1000 # 0.0031
np.maximum(0.7 * (label - ets_results[..., 2]), (1-0.7)*(ets_results[..., 2] -label )).mean()/1000 # 0.0051
np.maximum(0.5 * (label - ets_results[..., 1]), (1-0.5)*(ets_results[..., 1] -label )).mean()/1000  # 0.0059

np.mean(label < ets_results[..., 3]), 0.9 - np.mean(label < ets_results[..., 3])  ### 0.638
np.mean(label < ets_results[..., 2]), 0.7 - np.mean(label < ets_results[..., 2]) ### 0.640
np.mean(label < ets_results[..., 1]), 0.5- np.mean(label < ets_results[..., 1]) ### 0.894
#%%
theta_results = []
theta_forecaster = ThetaForecaster()    

theta_forecaster.fit(y_train)

for i in range(y_input.shape[0]):
    theta_forecaster.update(y_input[i], update_params=False)
    tmp_result = theta_forecaster.predict_quantiles(fh=fh, alpha=[0.1, 0.5, 0.7, 0.9])
    
    theta_results.append(tmp_result.values[np.newaxis, ...])
#%%
theta_results = np.concatenate(theta_results, axis=0)

np.maximum(0.9 * (label - theta_results[..., 3]), (1-0.9)*(theta_results[..., 3] -label )).mean()/1000 # 0.0031
np.maximum(0.7 * (label - theta_results[..., 2]), (1-0.7)*(theta_results[..., 2] -label )).mean()/1000 # 0.0051
np.maximum(0.5 * (label - theta_results[..., 1]), (1-0.5)*(theta_results[..., 1] -label )).mean()/1000  # 0.0059

np.mean(label < theta_results[..., 3]), 0.9 - np.mean(label < theta_results[..., 3])  ### 0.638
np.mean(label < theta_results[..., 2]), 0.7 - np.mean(label < theta_results[..., 2]) ### 0.640
np.mean(label < theta_results[..., 1]), 0.5- np.mean(label < theta_results[..., 1]) ### 0.894
#%%
import lightgbm as lgb 
from sktime.transformations.series.fourier import FourierFeatures
from sklearn.multioutput import MultiOutputRegressor
#%%
# def generate_ts_data_for_lgb(df, label_df, term_list, input_seq_len=48, tau=12):
#     conti_input_list = []
#     label_list = []
#     col_labels =  ['wl_1018680'] # ['wl_1018662', 'wl_1018680', 'wl_1018683', 'wl_1019630']
    
#     transformer = FourierFeatures(sp_list=[24], fourier_terms_list=[term_list])

#     for i in df['year'].unique():
#         tmp_df = np.array(df.loc[df['year'] == i, :])
#         tmp_label_df = np.array(label_df.loc[label_df['year'] == i, col_labels])
#         n = tmp_df.shape[0] - input_seq_len - tau 
#         covariate = np.zeros((n, input_seq_len, tmp_df.shape[1] - 4 + term_list*2))   
        
#         for j in range(n):
#             covariate[j, :, :] = np.concatenate([tmp_df[j:(j+input_seq_len), 4:], transformer.fit_transform(tmp_df[j:(j+input_seq_len), 15])], axis=1)
#             label[j, :, :] = tmp_label_df[(j+input_seq_len):(j+input_seq_len+tau), :]/1000

#         conti_input_list.append(covariate)
#         label_list.append(label)
    
#     total_conti_input = np.concatenate(conti_input_list, axis=0)
#     total_label = np.concatenate(label_list, axis=0)
    
#     return total_conti_input.reshape(-1, total_conti_input.shape[1] * total_conti_input.shape[2]), np.squeeze(total_label)
# #%%
# def generate_ts_eval_for_lgb(df, label_df, term_list, input_seq_len=48, tau=12):
#     conti_input_list = []
#     label_list = []
#     col_labels =  ['wl_1018680'] # ['wl_1018662', 'wl_1018680', 'wl_1018683', 'wl_1019630']
#     transformer = FourierFeatures(sp_list=[24], fourier_terms_list=[term_list])
    
#     tmp_df = np.array(df.loc[df['year'] == 2021, :])
#     tmp_label_df = np.array(label_df.loc[label_df['year'] == 2021, col_labels])
#     n = tmp_df.shape[0] - input_seq_len - tau 
    
#     covariate = np.zeros((n, input_seq_len, tmp_df.shape[1] - 4 + term_list * 2))     
#     label = np.zeros((n, tau, len(col_labels)))

#     for j in range(n):
#         covariate[j, :, :] = np.concatenate([tmp_df[j:(j+input_seq_len), 4:], transformer.fit_transform(tmp_df[j:(j+input_seq_len), 15])], axis=1)
#         label[j, :, :] = tmp_label_df[(j+input_seq_len):(j+input_seq_len+tau), :]/1000

#     conti_input_list.append(covariate)
#     label_list.append(label)
    
#     total_conti_input = np.concatenate(conti_input_list, axis=0)
#     total_label = np.concatenate(label_list, axis=0)
    
#     return total_conti_input.reshape(-1, total_conti_input.shape[1] * total_conti_input.shape[2]), np.squeeze(total_label) # label

#%%
def generate_ts_data_for_lgb(df, label_df, input_seq_len=48, tau=12):
    conti_input_list = []
    label_list = []
    col_labels =  ['wl_1018680'] # ['wl_1018662', 'wl_1018680', 'wl_1018683', 'wl_1019630']
    
    for i in df['year'].unique():
        tmp_df = np.array(df.loc[df['year'] == i, :])
        tmp_label_df = np.array(label_df.loc[label_df['year'] == i, col_labels])
        n = tmp_df.shape[0] - input_seq_len - tau 
        covariate = np.zeros((n, input_seq_len, tmp_df.shape[1] - 4))   
        label = np.zeros((n, tau, len(col_labels)))
        
        for j in range(n):
            covariate[j, :, :] = tmp_df[j:(j+input_seq_len), 4:]
            label[j, :, :] = tmp_label_df[(j+input_seq_len):(j+input_seq_len+tau), :]/1000

        conti_input_list.append(covariate)
        label_list.append(label)
    
    total_conti_input = np.concatenate(conti_input_list, axis=0)
    total_label = np.concatenate(label_list, axis=0)
    
    return total_conti_input.reshape(-1, total_conti_input.shape[1] * total_conti_input.shape[2]), np.squeeze(total_label)
#%%
def generate_ts_eval_for_lgb(df, label_df, input_seq_len=48, tau=12):
    conti_input_list = []
    label_list = []
    col_labels =  ['wl_1018680'] # ['wl_1018662', 'wl_1018680', 'wl_1018683', 'wl_1019630']
    # transformer = FourierFeatures(sp_list=[24], fourier_terms_list=[term_list])
    
    tmp_df = np.array(df.loc[df['year'] == 2021, :])
    tmp_label_df = np.array(label_df.loc[label_df['year'] == 2021, col_labels])
    n = tmp_df.shape[0] - input_seq_len - tau 
    
    covariate = np.zeros((n, input_seq_len, tmp_df.shape[1] - 4))     
    label = np.zeros((n, tau, len(col_labels)))

    for j in range(n):
        covariate[j, :, :] = tmp_df[j:(j+input_seq_len), 4:]
        label[j, :, :] = tmp_label_df[(j+input_seq_len):(j+input_seq_len+tau), :]/1000

    conti_input_list.append(covariate)
    label_list.append(label)
    
    total_conti_input = np.concatenate(conti_input_list, axis=0)
    total_label = np.concatenate(label_list, axis=0)
    
    return total_conti_input.reshape(-1, total_conti_input.shape[1] * total_conti_input.shape[2]), np.squeeze(total_label) # l

#%%
term_list = 4
transformer = FourierFeatures(sp_list=[24], fourier_terms_list=[term_list])
df_train_total_lgb = pd.concat([df_train_total, transformer.fit_transform(df_train_total[["11"]])], axis=1)
train_input_lgb, label_lgb = generate_ts_data_for_lgb(df_train_total_lgb, df_merged, input_seq_len=48, tau=12)
#%%
alphas = [0.5, 0.7, 0.9]

model_list = []

for alpha in alphas:
    params = {
        "objective": "quantile",
        "alpha": alpha,
        "boosting_type": "gbdt",
        # "max_bin": 50,
        # "num_leaves": 20,
        # "lambda_l1": 0.1,
        # "feature_fraction": 0.8
    }
    gbm = lgb.LGBMRegressor(**params)
    regr_multiglb = MultiOutputRegressor(gbm)
    regr_multiglb.fit(train_input_lgb, label_lgb)
    
    model_list.append(
        regr_multiglb
    )
#%%
df_test_total_lgb = pd.concat([df_test_total, transformer.fit_transform(df_test_total[["11"]])], axis=1)
test_input_lgb, test_label_lgb = generate_ts_eval_for_lgb(df_test_total_lgb, df_merged)
#%%
lgb_results_05 = model_list[0].predict(test_input_lgb)
lgb_results_07 = model_list[1].predict(test_input_lgb)
lgb_results_09 = model_list[2].predict(test_input_lgb)
#%%
np.maximum(0.9 * (test_label_lgb - lgb_results_09), (1-0.9)*(lgb_results_09 -test_label_lgb )).mean().round(4) 
np.maximum(0.7 * (test_label_lgb - lgb_results_07), (1-0.7)*(lgb_results_07 -test_label_lgb )).mean().round(4) 
np.maximum(0.5 * (test_label_lgb - lgb_results_05), (1-0.5)*(lgb_results_05 -test_label_lgb )).mean().round(4)   

np.mean(test_label_lgb < lgb_results_09).round(3), 0.9 - np.mean(test_label_lgb < lgb_results_09).round(3)  
np.mean(test_label_lgb < lgb_results_07).round(3), 0.7 - np.mean(test_label_lgb < lgb_results_07).round(3) 
np.mean(test_label_lgb < lgb_results_05).round(3), 0.5 - np.mean(test_label_lgb < lgb_results_05).round(3)
#%%
df_test_total
# feature_importances_05 = model_list[0].estimators_.feature_importances_

tmp_feature_importance_05 = np.zeros_like(model_list[0].estimators_[0].feature_importances_)

for i in range(len(model_list[0].estimators_)):
    tmp_feature_importance_05 =+ model_list[0].estimators_[i].feature_importances_

sorted_idx = tmp_feature_importance_05.argsort()[::-1]
sorted_idx[:5] 
tmp_feature_importance_05[sorted_idx[:5]]
# (tmp_feature_importance_05[sorted_idx[:5]]/12).round(2)
df_merged.columns[4:][(sorted_idx[:5]) % 24]
# df_merged.columns[4:][(sorted_idx[:5]) % 24]
# (sorted_idx[:5]) // 24

# tmp_feature_importance_05[sorted_idx[:5]]
# df_test_total_lgb.iloc[0:1, 4:]
#%%
# feature_importances_07 = model_list[1].estimators_.feature_importances_

tmp_feature_importance_07 = np.zeros_like(model_list[1].estimators_[0].feature_importances_)

for i in range(len(model_list[1].estimators_)):
    tmp_feature_importance_07 =+ model_list[1].estimators_[i].feature_importances_

sorted_idx = tmp_feature_importance_07.argsort()[::-1]
sorted_idx[:5] 

(tmp_feature_importance_07[sorted_idx[:5]]/12).round(2)

#%%
tmp_feature_importance_09 = np.zeros_like(model_list[2].estimators_[0].feature_importances_)

for i in range(len(model_list[2].estimators_)):
    tmp_feature_importance_09 =+ model_list[2].estimators_[i].feature_importances_

sorted_idx = tmp_feature_importance_09.argsort()[::-1]
sorted_idx[:5] 

(tmp_feature_importance_09[sorted_idx[:5]]/12).round(2)

df_merged.columns[4:][(sorted_idx[:5]) % 24]
(sorted_idx[:5]) // 24
#%% Classical distribution shift 
#### ARIMA
def generate_ds_arima(label_df, input_seq_len=48, tau=12):
    col_labels =  'wl_1018680' # ['wl_1018662', 'wl_1018680', 'wl_1018683', 'wl_1019630']
    
    tmp_df = np.array(label_df.loc[label_df['month'].isin([2,3]), col_labels])
    
    n = tmp_df.shape[0] - input_seq_len - tau 
    
    conti_input = np.zeros((n, input_seq_len), dtype=np.float32)
    label = np.zeros((n, tau))

    for j in range(n):
        conti_input[j, :] = tmp_df[j:(j+input_seq_len)]
        label[j, :] = tmp_df[(j+input_seq_len):(j+input_seq_len+tau)]

    return conti_input, label
#%%
year = "2016"

df_train_total = pd.read_csv("./data/df_train_total_ds_{}.csv".format(year))
df_test_total = pd.read_csv("./data/df_test_total_ds_{}.csv".format(year))
df_merged = pd.read_csv("./data/df_merged_ds_{}.csv".format(year))

fh = ForecastingHorizon(np.arange(1, 13), is_relative=True)

arima_forecaster = ARIMA()
# arima_forecaster = AutoARIMA(sp=12, d=0, max_p=2, max_q=2, suppress_warnings=True)
#%%
arima_results = []

y_train = df_merged.loc[df_merged["month"].isin([0, 1]), "wl_1018680"]
arima_forecaster.fit(y_train)

y_input, label = generate_ds_arima(df_merged)

for i in range(y_input.shape[0]):
    arima_forecaster.update(y_input[i], update_params=False)
    tmp_result = arima_forecaster.predict_quantiles(fh=fh, alpha=[0.1, 0.5, 0.7, 0.9])
    
    arima_results.append(tmp_result.values[np.newaxis, ...])
    print(i)
    # len(arima_results)
# %%
arima_results = np.concatenate(arima_results, axis=0)

print(np.maximum(0.9 * (label - arima_results[..., 3]), (1-0.9)*(arima_results[..., 3] -label )).mean()/1000) # 0.0031
print(np.mean(label < arima_results[..., 3]), 0.9 - np.mean(label < arima_results[..., 3]))  ### 0.638

print(np.maximum(0.7 * (label - arima_results[..., 2]), (1-0.7)*(arima_results[..., 2] -label )).mean()/1000) # 0.0051
print(np.mean(label < arima_results[..., 2]), 0.7 - np.mean(label < arima_results[..., 2])) ### 0.640

print(np.maximum(0.5 * (label - arima_results[..., 1]), (1-0.5)*(arima_results[..., 1] -label )).mean()/1000)  # 0.0059
print(np.mean(label < arima_results[..., 1]), 0.5- np.mean(label < arima_results[..., 1])) ### 0.894
#%% ETS
year = "2016"

df_train_total = pd.read_csv("./data/df_train_total_ds_{}.csv".format(year))
df_test_total = pd.read_csv("./data/df_test_total_ds_{}.csv".format(year))
df_merged = pd.read_csv("./data/df_merged_ds_{}.csv".format(year))

y_train = df_merged.loc[df_merged["month"].isin([0, 1]), "wl_1018680"]
arima_forecaster.fit(y_train)

ets_results = []
ets_forecaster = AutoETS(auto=True, n_jobs=-1)  

ets_forecaster.fit(y_train)

for i in range(y_input.shape[0]):
    ets_forecaster.update(y_input[i], update_params=False)
    tmp_result = ets_forecaster.predict_quantiles(fh=fh, alpha=[0.1, 0.5, 0.7, 0.9])
    
    ets_results.append(tmp_result.values[np.newaxis, ...])
    print(i)
#%%
ets_results = np.concatenate(ets_results, axis=0)

print(np.maximum(0.9 * (label - ets_results[..., 3]), (1-0.9)*(ets_results[..., 3] -label )).mean()/1000) # 0.0031
print(np.mean(label < ets_results[..., 3]), 0.9 - np.mean(label < ets_results[..., 3]))  ### 0.638

print(np.maximum(0.7 * (label - ets_results[..., 2]), (1-0.7)*(ets_results[..., 2] -label )).mean()/1000) # 0.0051
print(np.mean(label < ets_results[..., 2]), 0.7 - np.mean(label < ets_results[..., 2])) ### 0.640

print(np.maximum(0.5 * (label - ets_results[..., 1]), (1-0.5)*(ets_results[..., 1] -label )).mean()/1000)  # 0.0059
print(np.mean(label < ets_results[..., 1]), 0.5- np.mean(label < ets_results[..., 1])) ### 0.894

# %%
year = "2018"

df_train_total = pd.read_csv("./data/df_train_total_ds_{}.csv".format(year))
df_test_total = pd.read_csv("./data/df_test_total_ds_{}.csv".format(year))
df_merged = pd.read_csv("./data/df_merged_ds_{}.csv".format(year))

y_train = df_merged.loc[df_merged["month"].isin([0, 1]), "wl_1018680"]

theta_results = []
theta_forecaster = ThetaForecaster()    

theta_forecaster.fit(y_train)

y_input, label = generate_ds_arima(df_merged)

for i in range(y_input.shape[0]):
    theta_forecaster.update(y_input[i], update_params=False)
    tmp_result = theta_forecaster.predict_quantiles(fh=fh, alpha=[0.1, 0.5, 0.7, 0.9])
    
    theta_results.append(tmp_result.values[np.newaxis, ...])
#%%
theta_results = np.concatenate(theta_results, axis=0)

print(np.maximum(0.9 * (label - theta_results[..., 3]), (1-0.9)*(theta_results[..., 3] -label )).mean()/1000) # 0.0031
print(np.mean(label < theta_results[..., 3]), 0.9 - np.mean(label < theta_results[..., 3]))  ### 0.638

print(np.maximum(0.7 * (label - theta_results[..., 2]), (1-0.7)*(theta_results[..., 2] -label )).mean()/1000) # 0.0051
print(np.mean(label < theta_results[..., 2]), 0.7 - np.mean(label < theta_results[..., 2])) ### 0.640

print(np.maximum(0.5 * (label - theta_results[..., 1]), (1-0.5)*(theta_results[..., 1] -label )).mean()/1000)  # 0.0059
print(np.mean(label < theta_results[..., 1]), 0.5- np.mean(label < theta_results[..., 1])) ### 0.894
# %%
import lightgbm as lgb 
from sklearn.multioutput import MultiOutputRegressor

# def generate_ts_data_train_for_lgb(label_df, term_list, input_seq_len=48, tau=12):
#     conti_input_list = []
#     label_list = []
#     col_labels =  ['wl_1018680'] # ['wl_1018662', 'wl_1018680', 'wl_1018683', 'wl_1019630']
    
#     transformer = FourierFeatures(sp_list=[24], fourier_terms_list=[term_list])

#     for i in label_df['year'].unique():
#         tmp_df = np.array(label_df.loc[label_df['month'].isin([0, 1]), :])
#         tmp_label_df = np.array(label_df.loc[label_df['month'].isin([0, 1]), col_labels])        
#         n = tmp_df.shape[0] - input_seq_len - tau 
#         covariate = np.zeros((n, input_seq_len, tmp_df.shape[1] - 4 + term_list*2))   
#         label = np.zeros((n, tau, len(col_labels)))
        
#         for j in range(n):
#             covariate[j, :, :] = np.concatenate([tmp_df[j:(j+input_seq_len), 4:], transformer.fit_transform(tmp_df[j:(j+input_seq_len), 15])], axis=1)
#             label[j, :, :] = tmp_label_df[(j+input_seq_len):(j+input_seq_len+tau), :]/1000

#         conti_input_list.append(covariate)
#         label_list.append(label)
    
#     total_conti_input = np.concatenate(conti_input_list, axis=0)
#     total_label = np.concatenate(label_list, axis=0)
    
#     return total_conti_input.reshape(-1, total_conti_input.shape[1] * total_conti_input.shape[2]), np.squeeze(total_label)

# def generate_ts_data_test_for_lgb(label_df, term_list, input_seq_len=48, tau=12):
#     conti_input_list = []
#     label_list = []
#     col_labels =  ['wl_1018680'] # ['wl_1018662', 'wl_1018680', 'wl_1018683', 'wl_1019630']
#     transformer = FourierFeatures(sp_list=[24], fourier_terms_list=[term_list])
    
#     for i in label_df['year'].unique():
#         tmp_df = np.array(label_df.loc[label_df['month'].isin([2, 3]), :])
#         tmp_label_df = np.array(label_df.loc[label_df['month'].isin([2, 3]), col_labels])
#         n = tmp_df.shape[0] - input_seq_len - tau 
    
#         covariate = np.zeros((n, input_seq_len, tmp_df.shape[1] - 4 + term_list * 2))     
#         label = np.zeros((n, tau, len(col_labels)))

#         for j in range(n):
#             covariate[j, :, :] = np.concatenate([tmp_df[j:(j+input_seq_len), 4:], transformer.fit_transform(tmp_df[j:(j+input_seq_len), 15])], axis=1)
#             label[j, :, :] = tmp_label_df[(j+input_seq_len):(j+input_seq_len+tau), :]/1000

#         conti_input_list.append(covariate)
#         label_list.append(label)
    
#     total_conti_input = np.concatenate(conti_input_list, axis=0)
#     total_label = np.concatenate(label_list, axis=0)
    
#     return total_conti_input.reshape(-1, total_conti_input.shape[1] * total_conti_input.shape[2]), np.squeeze(total_label) # label
#%%
def generate_ts_data_train_for_lgb(label_df, term_list, input_seq_len=48, tau=12):
    conti_input_list = []
    label_list = []
    col_labels =  ['wl_1018680'] # ['wl_1018662', 'wl_1018680', 'wl_1018683', 'wl_1019630']
    
    transformer = FourierFeatures(sp_list=[24], fourier_terms_list=[term_list])

    for i in label_df['year'].unique():
        tmp_df = np.array(label_df.loc[label_df['month'].isin([0, 1]), :])
        tmp_label_df = np.array(label_df.loc[label_df['month'].isin([0, 1]), col_labels])        
        n = tmp_df.shape[0] - input_seq_len - tau 
        covariate = np.zeros((n, input_seq_len, tmp_df.shape[1] - 4 ))   
        label = np.zeros((n, tau, len(col_labels)))
        
        for j in range(n):
            covariate[j, :, :] = tmp_df[j:(j+input_seq_len), 4:]
            label[j, :, :] = tmp_label_df[(j+input_seq_len):(j+input_seq_len+tau), :]/1000

        conti_input_list.append(covariate)
        label_list.append(label)
    
    total_conti_input = np.concatenate(conti_input_list, axis=0)
    total_label = np.concatenate(label_list, axis=0)
    
    return total_conti_input.reshape(-1, total_conti_input.shape[1] * total_conti_input.shape[2]), np.squeeze(total_label)

def generate_ts_data_test_for_lgb(label_df, term_list, input_seq_len=48, tau=12):
    conti_input_list = []
    label_list = []
    col_labels =  ['wl_1018680'] # ['wl_1018662', 'wl_1018680', 'wl_1018683', 'wl_1019630']
    transformer = FourierFeatures(sp_list=[24], fourier_terms_list=[term_list])
    
    for i in label_df['year'].unique():
        tmp_df = np.array(label_df.loc[label_df['month'].isin([2, 3]), :])
        tmp_label_df = np.array(label_df.loc[label_df['month'].isin([2, 3]), col_labels])
        n = tmp_df.shape[0] - input_seq_len - tau 
    
        covariate = np.zeros((n, input_seq_len, tmp_df.shape[1] - 4 ))     
        label = np.zeros((n, tau, len(col_labels)))

        for j in range(n):
            covariate[j, :, :] = tmp_df[j:(j+input_seq_len), 4:]
            label[j, :, :] = tmp_label_df[(j+input_seq_len):(j+input_seq_len+tau), :]/1000

        conti_input_list.append(covariate)
        label_list.append(label)
    
    total_conti_input = np.concatenate(conti_input_list, axis=0)
    total_label = np.concatenate(label_list, axis=0)
    
    return total_conti_input.reshape(-1, total_conti_input.shape[1] * total_conti_input.shape[2]), np.squeeze(total_label) # label
#%%
year = "2016"

df_train_total = pd.read_csv("./data/df_train_total_ds_{}.csv".format(year))
df_test_total = pd.read_csv("./data/df_test_total_ds_{}.csv".format(year))
df_merged = pd.read_csv("./data/df_merged_ds_{}.csv".format(year))

# train_input_lgb, label_lgb = generate_ts_data_train_for_lgb(df_merged, term_list)
df_merged_lgb = pd.concat([df_merged, transformer.fit_transform(df_merged[["wl_1018680"]])], axis=1)
train_input_lgb, label_lgb = generate_ts_data_train_for_lgb(df_merged_lgb, term_list)
train_input_lgb.shape
label_lgb.shape
#%%
alphas = [0.5, 0.7, 0.9]

model_list = []

for alpha in alphas:
    params = {
        "objective": "quantile",
        "alpha": alpha,
        "boosting": "gbdt",

    }
    gbm = lgb.LGBMRegressor(**params)
    regr_multiglb = MultiOutputRegressor(gbm)
    regr_multiglb.fit(train_input_lgb, label_lgb)
    
    model_list.append(
        regr_multiglb
    )
# %%
test_input_lgb, test_label_lgb = generate_ts_data_test_for_lgb(df_merged_lgb, term_list)
test_input_lgb.shape
#%%
lgb_results_05 = model_list[0].predict(test_input_lgb)
lgb_results_07 = model_list[1].predict(test_input_lgb)
lgb_results_09 = model_list[2].predict(test_input_lgb)
#%%
print(round(np.maximum(0.9 * (test_label_lgb - lgb_results_09), (1-0.9)*(lgb_results_09 -test_label_lgb )).mean(), 4)) # 0.0031
print(round(np.mean(test_label_lgb < lgb_results_09), 4), round(0.9 - np.mean(test_label_lgb < lgb_results_09),4))  ### 0.638

print(round(np.maximum(0.7 * (test_label_lgb - lgb_results_07), (1-0.7)*(lgb_results_07 -test_label_lgb )).mean(), 4)) # 0.0051
print(round(np.mean(test_label_lgb < lgb_results_07),4), round(0.7 - np.mean(test_label_lgb < lgb_results_07), 4)) ### 0.640

print(round(np.maximum(0.5 * (test_label_lgb - lgb_results_05), (1-0.5)*(lgb_results_05 -test_label_lgb )).mean(), 4))   # 0.0059
print(round(np.mean(test_label_lgb < lgb_results_05), 4), round(0.5- np.mean(test_label_lgb < lgb_results_05),4)) ### 0.894
# %%
