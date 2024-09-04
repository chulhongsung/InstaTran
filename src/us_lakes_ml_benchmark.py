import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

os.chdir(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

df_mead = pd.read_csv("../data/us_lakes/df_mead_preprocessed.csv")
df_mohave = pd.read_csv("../data/us_lakes/df_mohave_preprocessed.csv")
df_havasu = pd.read_csv("../data/us_lakes/df_havasu_preprocessed.csv")

def train_valid_test_split(df_mead, df_mohave, df_havasu, valid_size=2/9, test_size=1/3):
    
    N, P = df_mohave.shape
    
    df_mead.columns = ["e1", "i1", "o1", "y", "m", "d"]
    df_mohave.columns = ["e2", "i2", "o2", "y", "m", "d", "p1"]
    df_havasu.columns = ["e3", "i3", "o3", "y", "m", "d", "p2"]
        
    index_1 = round(N * (1 - valid_size - test_size))
    index_2 = round(N * (1-test_size))
    
    df_mead_train = df_mead.iloc[:index_1, :]
    df_mohave_train = df_mohave.iloc[:index_1, :]
    df_havasu_train = df_havasu.iloc[:index_1, :]
    
    df_mead_valid = df_mead.iloc[index_1:index_2, :]
    df_mohave_valid = df_mohave.iloc[index_1:index_2, :]
    df_havasu_valid = df_havasu.iloc[index_1:index_2, :]
    
    df_mead_test = df_mead.iloc[index_2:, :]
    df_mohave_test = df_mohave.iloc[index_2:, :]
    df_havasu_test = df_havasu.iloc[index_2:, :]
    
    return (df_mead_train, df_mohave_train, df_havasu_train), (df_mead_valid, df_mohave_valid, df_havasu_valid), (df_mead_test, df_mohave_test, df_havasu_test)

train, valid, test = train_valid_test_split(df_mead.loc[df_mead["year"] <= 2013],
                                            df_mohave.loc[df_mohave["year"] <= 2013],
                                            df_havasu.loc[df_havasu["year"] <= 2013])


from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.arima import ARIMA, AutoARIMA
from sktime.forecasting.theta import ThetaForecaster

def generate_eval_arima(label_df, input_seq_len=24, tau=4):
    col_labels = "e3" # ['wl_1018662', 'wl_1018680', 'wl_1018683', 'wl_1019630']
    
    tmp_df = np.array(label_df.loc[:, col_labels])
    
    n = tmp_df.shape[0] - input_seq_len - tau 
    
    conti_input = np.zeros((n, input_seq_len), dtype=np.float32)
    label = np.zeros((n, tau))

    for j in range(n):
        conti_input[j, :] = tmp_df[j:(j+input_seq_len)]
        label[j, :] = tmp_df[(j+input_seq_len):(j+input_seq_len+tau)]

    return conti_input, label

data_split_range = [(2005, 2013), (2008, 2016), (2011, 2019), (2014, 2022)]

ql_09 = []
ql_07 = []
ql_05 = []
ql_03 = []
ql_01 = []

qr_09 = []
qr_07 = []
qr_05 = []
qr_03 = []
qr_01 = []

for a, b in data_split_range:
    tmp_train, tmp_valid, tmp_test = train_valid_test_split(df_mead.loc[(df_mead["year"] >= a) & (df_mead["year"] <= b)],
                                            df_mohave.loc[(df_mohave["year"] >= a) & (df_mohave["year"] <= b)],
                                            df_havasu.loc[(df_havasu["year"] >= a) & (df_havasu["year"] <= b)])
     
    fh = ForecastingHorizon(np.arange(1, 5), is_relative=True)

    ets_results = []
    ets_forecaster = AutoETS(auto=True, n_jobs=-1)  

    ets_forecaster.fit(tmp_train[2]["e3"])
    
    y_test_input, label = generate_eval_arima(tmp_test[2]) 
    
    for i in range(y_test_input.shape[0]):
        ets_forecaster.update(y_test_input[i], update_params=False)
        tmp_result = ets_forecaster.predict_quantiles(fh=fh, alpha=[0.1, 0.3, 0.5, 0.7, 0.9])
        
        ets_results.append(tmp_result.values[np.newaxis, ...])
        
    ets_results = np.concatenate(ets_results, axis=0)

    ql_09.append(np.maximum(0.9 * (label - ets_results[..., 4]), (1-0.9)*(ets_results[..., 4] - label)).mean())
    ql_07.append(np.maximum(0.7 * (label - ets_results[..., 3]), (1-0.7)*(ets_results[..., 3] - label)).mean())
    ql_05.append(np.maximum(0.5 * (label - ets_results[..., 2]), (1-0.5)*(ets_results[..., 2] - label)).mean())
    ql_03.append(np.maximum(0.3 * (label - ets_results[..., 1]), (1-0.3)*(ets_results[..., 1] - label)).mean())
    ql_01.append(np.maximum(0.1 * (label - ets_results[..., 0]), (1-0.1)*(ets_results[..., 0] - label)).mean())
        
    qr_09.append((np.mean(label < ets_results[..., 4]), 0.9 - np.mean(label < ets_results[..., 4])))
    qr_07.append((np.mean(label < ets_results[..., 3]), 0.7 - np.mean(label < ets_results[..., 3])))
    qr_05.append((np.mean(label < ets_results[..., 2]), 0.5 - np.mean(label < ets_results[..., 2])))
    qr_03.append((np.mean(label < ets_results[..., 1]), 0.3 - np.mean(label < ets_results[..., 1])))
    qr_01.append((np.mean(label < ets_results[..., 0]), 0.1 - np.mean(label < ets_results[..., 0])))

np.array(ql_09).mean().round(3)
np.array(ql_05).mean().round(3)
np.array(ql_01).mean().round(3)

np.array([x for x, _ in qr_09]).mean().round(3)
np.array([np.abs(x) for _, x in qr_09]).mean().round(3)

np.array([x for x, _ in qr_05]).mean().round(3)
np.array([np.abs(x) for _, x in qr_05]).mean().round(3)

np.array([x for x, _ in qr_01]).mean().round(3)
np.array([np.abs(x) for _, x in qr_01]).mean().round(3)
# np.array([np.abs(0.1 - x) for x,_  in qr_01]).mean().round(3)

ql_09 = []
ql_07 = []
ql_05 = []
ql_03 = []
ql_01 = []

qr_09 = []
qr_07 = []
qr_05 = []
qr_03 = []
qr_01 = []

for a, b in data_split_range:
    tmp_train, tmp_valid, tmp_test = train_valid_test_split(df_mead.loc[(df_mead["year"] >= a) & (df_mead["year"] <= b)],
                                            df_mohave.loc[(df_mohave["year"] >= a) & (df_mohave["year"] <= b)],
                                            df_havasu.loc[(df_havasu["year"] >= a) & (df_havasu["year"] <= b)])
     
    fh = ForecastingHorizon(np.arange(1, 5), is_relative=True)

    tmp_results = []
    forecaster = AutoARIMA()  

    forecaster.fit(tmp_train[2]["e3"])
    
    y_test_input, label = generate_eval_arima(tmp_test[2]) 
    
    for i in range(y_test_input.shape[0]):
        forecaster.update(y_test_input[i], update_params=False)
        tmp_result = forecaster.predict_quantiles(fh=fh, alpha=[0.1, 0.3, 0.5, 0.7, 0.9])
        
        tmp_results.append(tmp_result.values[np.newaxis, ...])
        
    tmp_results = np.concatenate(tmp_results, axis=0)
    
    ql_09.append(np.maximum(0.9 * (label - tmp_results[..., 4]), (1-0.9)*(tmp_results[..., 4] - label)).mean())
    ql_07.append(np.maximum(0.7 * (label - tmp_results[..., 3]), (1-0.7)*(tmp_results[..., 3] - label)).mean())
    ql_05.append(np.maximum(0.5 * (label - tmp_results[..., 2]), (1-0.5)*(tmp_results[..., 2] - label)).mean())
    ql_03.append(np.maximum(0.3 * (label - tmp_results[..., 1]), (1-0.3)*(tmp_results[..., 1] - label)).mean())
    ql_01.append(np.maximum(0.1 * (label - tmp_results[..., 0]), (1-0.1)*(tmp_results[..., 0] - label)).mean())
        
    qr_09.append((np.mean(label < tmp_results[..., 4]), 0.9 - np.mean(label < tmp_results[..., 4])))
    qr_07.append((np.mean(label < tmp_results[..., 3]), 0.7 - np.mean(label < tmp_results[..., 3])))
    qr_05.append((np.mean(label < tmp_results[..., 2]), 0.5 - np.mean(label < tmp_results[..., 2])))
    qr_03.append((np.mean(label < tmp_results[..., 1]), 0.3 - np.mean(label < tmp_results[..., 1])))
    qr_01.append((np.mean(label < tmp_results[..., 0]), 0.1 - np.mean(label < tmp_results[..., 0])))

np.array(ql_09).mean().round(3)
np.array(ql_05).mean().round(3)
np.array(ql_01).mean().round(3)

np.array([x for x, _ in qr_09]).mean().round(3)
np.array([np.abs(x) for _, x in qr_09]).mean().round(3)

np.array([x for x, _ in qr_05]).mean().round(3)
np.array([np.abs(x) for _, x in qr_05]).mean().round(3)

np.array([x for x, _ in qr_01]).mean().round(3)
np.array([np.abs(x) for _, x in qr_01]).mean().round(3)

ql_09 = []
ql_07 = []
ql_05 = []
ql_03 = []
ql_01 = []

qr_09 = []
qr_07 = []
qr_05 = []
qr_03 = []
qr_01 = []

for a, b in data_split_range:
    tmp_train, tmp_valid, tmp_test = train_valid_test_split(df_mead.loc[(df_mead["year"] >= a) & (df_mead["year"] <= b)],
                                            df_mohave.loc[(df_mohave["year"] >= a) & (df_mohave["year"] <= b)],
                                            df_havasu.loc[(df_havasu["year"] >= a) & (df_havasu["year"] <= b)])
     
    fh = ForecastingHorizon(np.arange(1, 5), is_relative=True)

    tmp_results = []
    forecaster = ThetaForecaster()  

    forecaster.fit(tmp_train[2]["e3"])
    
    y_test_input, label = generate_eval_arima(tmp_test[2]) 
    
    for i in range(y_test_input.shape[0]):
        forecaster.update(y_test_input[i], update_params=False)
        tmp_result = forecaster.predict_quantiles(fh=fh, alpha=[0.1, 0.3, 0.5, 0.7, 0.9])
        
        tmp_results.append(tmp_result.values[np.newaxis, ...])
        
    tmp_results = np.concatenate(tmp_results, axis=0)
    
    ql_09.append(np.maximum(0.9 * (label - tmp_results[..., 4]), (1-0.9)*(tmp_results[..., 4] - label)).mean())
    ql_07.append(np.maximum(0.7 * (label - tmp_results[..., 3]), (1-0.7)*(tmp_results[..., 3] - label)).mean())
    ql_05.append(np.maximum(0.5 * (label - tmp_results[..., 2]), (1-0.5)*(tmp_results[..., 2] - label)).mean())
    ql_03.append(np.maximum(0.3 * (label - tmp_results[..., 1]), (1-0.3)*(tmp_results[..., 1] - label)).mean())
    ql_01.append(np.maximum(0.1 * (label - tmp_results[..., 0]), (1-0.1)*(tmp_results[..., 0] - label)).mean())
        
    qr_09.append((np.mean(label < tmp_results[..., 4]), 0.9 - np.mean(label < tmp_results[..., 4])))
    qr_07.append((np.mean(label < tmp_results[..., 3]), 0.7 - np.mean(label < tmp_results[..., 3])))
    qr_05.append((np.mean(label < tmp_results[..., 2]), 0.5 - np.mean(label < tmp_results[..., 2])))
    qr_03.append((np.mean(label < tmp_results[..., 1]), 0.3 - np.mean(label < tmp_results[..., 1])))
    qr_01.append((np.mean(label < tmp_results[..., 0]), 0.1 - np.mean(label < tmp_results[..., 0])))


np.array(ql_09).mean().round(3)
np.array(ql_05).mean().round(3)
np.array(ql_01).mean().round(3)

np.array([x for x, _ in qr_09]).mean().round(3)
np.array([np.abs(x) for _, x in qr_09]).mean().round(3)

np.array([x for x, _ in qr_05]).mean().round(3)
np.array([np.abs(x) for _, x in qr_05]).mean().round(3)

np.array([x for x, _ in qr_01]).mean().round(3)
np.array([np.abs(x) for _, x in qr_01]).mean().round(3)

import lightgbm as lgb 
from sktime.transformations.series.fourier import FourierFeatures
from sklearn.multioutput import MultiOutputRegressor

def generate_data_for_lgb(df_lakes, term_list, scaler=None, input_seq_len=24, tau=4):
    col_labels = "e3" 
    

    tmp_arr = np.array(df_lakes.drop(columns=['y', 'm', 'd']))
    
    tmp_cate = np.array(df_lakes[["y", "m", "d"]])
    
    if not scaler:
        scaler = MinMaxScaler()
        tmp_arr = scaler.fit_transform(tmp_arr)

    if scaler:
        tmp_arr = scaler.fit_transform(tmp_arr)
        
    tmp_label = np.array(df_lakes.loc[:, col_labels])
    _, p = tmp_arr.shape
    n = tmp_arr.shape[0] - input_seq_len - tau 
    
    input = np.zeros((n, input_seq_len, p + term_list * 2), dtype=np.float32)
    cate_input = np.zeros((n, input_seq_len, 3), dtype=np.int32)
    label = np.zeros((n, tau))
    
    transformer = FourierFeatures(sp_list=[30], fourier_terms_list=[term_list])
    
    for j in range(n):
        input[j, :] = np.concatenate([tmp_arr[j:(j+input_seq_len)], transformer.fit_transform(tmp_label[j:(j+input_seq_len)])], axis=1)
        cate_input[j, :] = tmp_cate[j:(j+input_seq_len)]
        label[j, :] = tmp_label[(j+input_seq_len):(j+input_seq_len+tau)]

    return input.reshape(n, input_seq_len * (p + term_list * 2)), cate_input.reshape(n, -1), label, scaler 



term_list = 2

data_split_range = [(2005, 2013), (2008, 2016), (2011, 2019), (2014, 2022)]

ql_09 = []
ql_07 = []
ql_05 = []
ql_03 = []
ql_01 = []

qr_09 = []
qr_07 = []
qr_05 = []
qr_03 = []
qr_01 = []

for a, b in data_split_range:
    tmp_train, tmp_valid, tmp_test = train_valid_test_split(df_mead.loc[(df_mead["year"] >= a) & (df_mead["year"] <= b)],
                                            df_mohave.loc[(df_mohave["year"] >= a) & (df_mohave["year"] <= b)],
                                            df_havasu.loc[(df_havasu["year"] >= a) & (df_havasu["year"] <= b)])
    
    train_df_lakes = pd.concat([tmp_train[0].drop(columns=['y', 'm', 'd']), tmp_train[1].drop(columns=['y', 'm', 'd']), tmp_train[2]], axis=1)
    
    valid_df_lakes = pd.concat([tmp_valid[0].drop(columns=['y', 'm', 'd']), tmp_valid[1].drop(columns=['y', 'm', 'd']), tmp_valid[2]], axis=1)
    
    train_input_lgb, train_cate_lgb, train_label_lgb, scaler = generate_data_for_lgb(train_df_lakes, term_list=term_list)
    
    valid_input_lgb, valid_cate_lgb, valid_label_lgb, _ = generate_data_for_lgb(valid_df_lakes, scaler=scaler, term_list=term_list)
    
    test_df_lakes = pd.concat([tmp_test[0].drop(columns=['y', 'm', 'd']), tmp_test[1].drop(columns=['y', 'm', 'd']), tmp_test[2]], axis=1)
    
    test_input_lgb, test_cate_lgb, test_label_lgb, _ = generate_data_for_lgb(test_df_lakes, term_list=term_list, scaler=scaler)
    
    alphas = [0.1, 0.5, 0.9]

    model_list = []

    for alpha in alphas:
        params = {
            "objective": "quantile",
            "alpha": alpha,
            "num_leaves": 20, 
            "boosting_type": "gbdt",
            "categorical_feature": np.arange(312, 384),
            "eval_set": [np.concatenate([valid_input_lgb, valid_cate_lgb], axis=1), valid_label_lgb],
            "eval_metric": "quantile",
            "verbose": 0,
            "reg_alpha": 1.,
            "reg_lambda" : 1.,
            "n_estimators": 5
        }
        gbm = lgb.LGBMRegressor(**params)
        regr_multiglb = MultiOutputRegressor(gbm)
        regr_multiglb.fit(np.concatenate([train_input_lgb, train_cate_lgb], axis=1), train_label_lgb)
        
        model_list.append(
            regr_multiglb
        )
    
    lgb_results_01 = model_list[0].predict(np.concatenate([test_input_lgb, test_cate_lgb], axis=1))
    #lgb_results_03 = model_list[1].predict(test_input_lgb)
    lgb_results_05 = model_list[1].predict(np.concatenate([test_input_lgb, test_cate_lgb], axis=1))
    #lgb_results_07 = model_list[3].predict(test_input_lgb)
    lgb_results_09 = model_list[2].predict(np.concatenate([test_input_lgb, test_cate_lgb], axis=1))

    
    ql_09.append(np.maximum(0.9 * (test_label_lgb - lgb_results_09), (1-0.9)*(lgb_results_09 - test_label_lgb)).mean())
    #ql_07.append(np.maximum(0.7 * (test_label_lgb - lgb_results_07), (1-0.7)*(lgb_results_07 - test_label_lgb)).mean())
    ql_05.append(np.maximum(0.5 * (test_label_lgb - lgb_results_05), (1-0.5)*(lgb_results_05 - test_label_lgb)).mean())
    #ql_03.append(np.maximum(0.3 * (test_label_lgb - lgb_results_03), (1-0.3)*(lgb_results_03 - test_label_lgb)).mean())
    ql_01.append(np.maximum(0.1 * (test_label_lgb - lgb_results_01), (1-0.1)*(lgb_results_01 - test_label_lgb)).mean())
        
    qr_09.append((np.mean(test_label_lgb < lgb_results_09), 0.9 - np.mean(test_label_lgb < lgb_results_09)))
    #qr_07.append((np.mean(test_label_lgb < lgb_results_07), 0.7 - np.mean(test_label_lgb < lgb_results_07)))
    qr_05.append((np.mean(test_label_lgb < lgb_results_05), 0.5 - np.mean(test_label_lgb < lgb_results_05)))
    #qr_03.append((np.mean(test_label_lgb < lgb_results_03), 0.3 - np.mean(test_label_lgb < lgb_results_03)))
    qr_01.append((np.mean(test_label_lgb < lgb_results_01), 0.1 - np.mean(test_label_lgb < lgb_results_01)))

np.array(ql_09).mean().round(3)
np.array(ql_05).mean().round(3)
np.array(ql_01).mean().round(3)

np.array([x for x, _ in qr_09]).mean().round(3)
np.array([np.abs(x) for _, x in qr_09]).mean().round(3)

np.array([x for x, _ in qr_05]).mean().round(3)
np.array([np.abs(x) for _, x in qr_05]).mean().round(3)

np.array([x for x, _ in qr_01]).mean().round(3)
np.array([np.abs(x) for _, x in qr_01]).mean().round(3)