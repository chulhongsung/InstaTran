import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset #IterableDataset

import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from utils import *
from layers import *
from models import DeepAR, MQRnn, TemporalFusionTransformer, STALSTM

from tqdm import tqdm
import pandas as pd
import numpy as np 

from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

os.chdir(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
df_mead = pd.read_csv("../data/us_lakes/df_mead_preprocessed.csv")
df_mohave = pd.read_csv("../data/us_lakes/df_mohave_preprocessed.csv")
df_havasu = pd.read_csv("../data/us_lakes/df_havasu_preprocessed.csv")


def train_valid_test_split_for_dl(df_mead, df_mohave, df_havasu, valid_size=2/9, test_size=1/3, input_seq_len=24, tau=4):
    N, _ = df_mohave.shape
    
    scaler = MinMaxScaler()
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    
    df_mead.columns = ["e1", "i1", "o1", "y", "m", "d"]
    df_mohave.columns = ["e2", "i2", "o2", "y", "m", "d", "p1"]
    df_havasu.columns = ["e3", "i3", "o3", "y", "m", "d", "p2"]
    
    df_mead.drop(columns=["y", "m", "d"], inplace=True)
    df_mohave.drop(columns=["y", "m", "d"], inplace=True)
    df_havasu.drop(columns=["y"], inplace=True)
    
    df_havasu_date = df_havasu.loc[:, ["m", "d"]]
    df_havasu_cov = df_havasu.loc[:, ["e3", "i3", "o3", "p2"]]
    
    index_1 = round(N * (1 - valid_size - test_size))
    index_2 = round(N * (1-test_size))
    
    df_mead_train = df_mead.iloc[:index_1, :]
    df_mohave_train = df_mohave.iloc[:index_1, :]
    df_havasu_train = df_havasu_cov.iloc[:index_1, :]
    df_train_date = df_havasu_date.iloc[:index_1, :]
    
    df_mead_valid = df_mead.iloc[index_1:index_2, :]
    df_mohave_valid = df_mohave.iloc[index_1:index_2, :]
    df_havasu_valid = df_havasu_cov.iloc[index_1:index_2, :]
    df_valid_date = df_havasu_date.iloc[index_1:index_2, :]

    df_mead_test = df_mead.iloc[index_2:, :]
    df_mohave_test = df_mohave.iloc[index_2:, :]
    df_havasu_test = df_havasu_cov.iloc[index_2:, :]
    df_test_date = df_havasu_date.iloc[index_2:, :]
    
    df_train = pd.concat([df_mead_train, df_mohave_train, df_havasu_train], axis=1)
    df_valid = pd.concat([df_mead_valid, df_mohave_valid, df_havasu_valid], axis=1)
    df_test = pd.concat([df_mead_test, df_mohave_test, df_havasu_test], axis=1)
    
    imp_mean.fit(df_train)
    df_train = imp_mean.transform(df_train)
    df_valid = imp_mean.transform(df_valid)
    df_test = imp_mean.transform(df_test)
    
    col_labels = 7
    
    tmp_arr = np.array(df_train)
    tmp_label = np.array(df_train[:, col_labels])
    tmp_date = np.array(df_train_date)
    
    _, p = tmp_arr.shape
    n = tmp_arr.shape[0] - input_seq_len - tau 
    
    train_input = np.zeros((n, input_seq_len, p), dtype=np.float32)
    train_label = np.zeros((n, tau))
    train_date = np.zeros((n, input_seq_len, 2))
        
    for j in range(n):
        train_input[j, :] = tmp_arr[j:(j+input_seq_len)]
        train_label[j, :] = tmp_label[(j+input_seq_len):(j+input_seq_len+tau)]/1000
        train_date[j, :] = tmp_date[j:(j+input_seq_len)]

    tmp_arr = np.array(df_valid)
    tmp_label = np.array(df_valid[:, col_labels])
    tmp_date = np.array(df_valid_date)
    
    _, p = tmp_arr.shape
    n = tmp_arr.shape[0] - input_seq_len - tau 
    
    valid_input = np.zeros((n, input_seq_len, p), dtype=np.float32)
    valid_label = np.zeros((n, tau))
    valid_date = np.zeros((n, input_seq_len, 2))
    
    for j in range(n):
        valid_input[j, :] = tmp_arr[j:(j+input_seq_len)]
        valid_label[j, :] = tmp_label[(j+input_seq_len):(j+input_seq_len+tau)]/1000
        valid_date[j, :] = tmp_date[j:(j+input_seq_len)]
        
    tmp_arr = np.array(df_test)
    tmp_label = np.array(df_test[:, col_labels])
    tmp_date = np.array(df_test_date)
    
    _, p = tmp_arr.shape
    n = tmp_arr.shape[0] - input_seq_len - tau 
    
    test_input = np.zeros((n, input_seq_len, p), dtype=np.float32)
    test_label = np.zeros((n, tau))
    test_date = np.zeros((n, input_seq_len, 2))
    
    for j in range(n):
        test_input[j, :] = tmp_arr[j:(j+input_seq_len)]
        test_label[j, :] = tmp_label[(j+input_seq_len):(j+input_seq_len+tau)]/1000
        test_date[j, :] = tmp_date[j:(j+input_seq_len)]
    
    
    scaler.fit(train_input.reshape(train_input.shape[0], -1))
    train_scaled = scaler.transform(train_input.reshape(train_input.shape[0], -1)).reshape(train_input.shape[0], input_seq_len, -1)
    valid_scaled = scaler.transform(valid_input.reshape(valid_input.shape[0], -1)).reshape(valid_input.shape[0], input_seq_len, -1)
    test_scaled = scaler.transform(test_input.reshape(test_input.shape[0], -1)).reshape(test_input.shape[0], input_seq_len, -1)
    
    return (train_scaled, train_date, train_label), (valid_scaled, valid_date, valid_label), (test_scaled, test_date, test_label), scaler

# %%
train_set, valid_set, test_set, scaler = train_valid_test_split_for_dl(df_mead.loc[df_mead["year"] <= 2013],
                                            df_mohave.loc[df_mohave["year"] <= 2013],
                                            df_havasu.loc[df_havasu["year"] <= 2013])
# %%
train_dataset = TensorDataset(torch.FloatTensor(train_set[0]), torch.FloatTensor(train_set[2]))
# test_dataset = TensorDataset(torch.FloatTensor(test_conti_input), torch.LongTensor(test_cate_input), torch.LongTensor(test_future_input), torch.FloatTensor(test_label))
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=256)

class SpatialAttention(nn.Module):
    def __init__(self, num_feature):
        super(SpatialAttention, self).__init__()
        self.linear = nn.Linear(num_feature, num_feature)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
            
    def forward(self, x):
        alpha = self.softmax(self.sigmoid(self.linear(x)))
        return x * alpha, alpha

class TemporalAttention(nn.Module):
    def __init__(self, num_feature):
        super(TemporalAttention, self).__init__()
        self.linear = nn.Linear(num_feature, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-2)
            
    def forward(self, x):
        beta = self.softmax(self.relu(self.linear(x)))
        return (x * beta).sum(axis=-2), beta

class STALSTM(nn.Module):
    """Ding et al., 2020 (Neurocomputing)

    Args:
        nn (_type_): _description_
    """
    def __init__(self, d_model, num_feature, tau, num_quantiles):
        super(STALSTM, self).__init__()
        self.d_model = d_model
        self.sa = SpatialAttention(num_feature)
        self.lstm = nn.LSTM(input_size=num_feature, hidden_size=d_model, batch_first=True)
        self.ta = TemporalAttention(d_model)
        self.qol = nn.ModuleList([nn.Linear(d_model, tau) for _ in range(num_quantiles)])

    def forward(self, x):
        x_, alpha = self.sa(x)
        h, (_, _) = self.lstm(x_)
        h_, beta = self.ta(h)
        
        total_output_list = []
        
        for _,l in enumerate(self.qol):
            tmp_quantile_output = l(h_)
            total_output_list.append(tmp_quantile_output.unsqueeze(-1))
        
        return torch.cat(total_output_list, dim=-1), alpha, beta
    
#%%
ding = STALSTM(12, 11, 4, 5)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ding.to(device)
optimizer = optim.Adam(ding.parameters(), lr=0.001)

class QuantileRisk(nn.Module):
    def __init__(self, tau, quantile, device):
        super(QuantileRisk, self).__init__()
        self.quantile = quantile
        self.device = device
        self.q_arr = torch.tensor(quantile).float().unsqueeze(-1).repeat(1, 1, tau).transpose(-1, -2).to(self.device)
    
    def forward(self, true, pred):
        
        ql = torch.maximum(self.q_arr * (true.unsqueeze(-1) - pred), (1-self.q_arr)*(pred - true.unsqueeze(-1)))

        return ql.mean()

def train(model, loader, criterion, optimizer, device):
    
    model.train()
    
    total_loss = []
    
    for batch in loader:
        conti_input, true_y = batch 
        
        conti_input = conti_input.to(device)
        true_y = true_y.to(device)
        
        pred, _, _ = model(conti_input)
        
        loss = criterion(true_y, pred)
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        total_loss.append(loss)
        
    return sum(total_loss)/len(total_loss)

criterion = QuantileRisk(4, [0.1, 0.3, 0.5, 0.7, 0.9], device)
#%%
pbar = tqdm(range(100))

for epoch in pbar:        
    train_loss = train(ding, train_loader, criterion, optimizer, device)
    pbar.set_description("Train Loss: {:.4f}".format(train_loss))
    
torch.save(ding.state_dict(), './assets/weights/us_lakes_ding.pth')
# %%
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
    tmp_train, tmp_valid, tmp_test, scaler = train_valid_test_split_for_dl(df_mead.loc[(df_mead["year"] >= a) & (df_mead["year"] <= b)],
                                            df_mohave.loc[(df_mohave["year"] >= a) & (df_mohave["year"] <= b)],
                                            df_havasu.loc[(df_havasu["year"] >= a) & (df_havasu["year"] <= b)])
     
    train_dataset = TensorDataset(torch.FloatTensor(tmp_train[0]), torch.FloatTensor(tmp_train[2]))
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=256)

    ding = STALSTM(12, 11, 4, 5)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ding.to(device)
    optimizer = optim.Adam(ding.parameters(), lr=0.001)
    criterion = QuantileRisk(4, [0.1, 0.3, 0.5, 0.7, 0.9], device)
    
    pbar = tqdm(range(200))

    for epoch in pbar:        
        train_loss = train(ding, train_loader, criterion, optimizer, device)
        pbar.set_description("Train Loss: {:.4f}".format(train_loss))
        
    test_input = torch.FloatTensor(tmp_test[0])
    label = tmp_test[2]
    
    ding.eval()    
    with torch.no_grad():
        pred_results, _, _ = ding(test_input)
        pred_results = pred_results.detach().cpu().numpy()
    
    ql_09.append(np.maximum(0.9 * (label - pred_results[..., 4]), (1-0.9)*(pred_results[..., 4] - label)).mean() * 1000)
    ql_07.append(np.maximum(0.7 * (label - pred_results[..., 3]), (1-0.7)*(pred_results[..., 3] - label)).mean() * 1000)
    ql_05.append(np.maximum(0.5 * (label - pred_results[..., 2]), (1-0.5)*(pred_results[..., 2] - label)).mean() * 1000)
    ql_03.append(np.maximum(0.3 * (label - pred_results[..., 1]), (1-0.3)*(pred_results[..., 1] - label)).mean() * 1000)
    ql_01.append(np.maximum(0.1 * (label - pred_results[..., 0]), (1-0.1)*(pred_results[..., 0] - label)).mean() * 1000)
        
    qr_09.append((np.mean(label < pred_results[..., 4]), 0.9 - np.mean(label < pred_results[..., 4])))
    qr_07.append((np.mean(label < pred_results[..., 3]), 0.7 - np.mean(label < pred_results[..., 3])))
    qr_05.append((np.mean(label < pred_results[..., 2]), 0.5 - np.mean(label < pred_results[..., 2])))
    qr_03.append((np.mean(label < pred_results[..., 1]), 0.3 - np.mean(label < pred_results[..., 1])))
    qr_01.append((np.mean(label < pred_results[..., 0]), 0.1 - np.mean(label < pred_results[..., 0])))

np.array(ql_09).mean().round(3)
np.array(ql_05).mean().round(3)
np.array(ql_01).mean().round(3)

np.array([x for x, _ in qr_09]).mean().round(3)
np.array([np.abs(x) for _, x in qr_09]).mean().round(3)

np.array([x for x, _ in qr_05]).mean().round(3)
np.array([np.abs(x) for _, x in qr_05]).mean().round(3)

np.array([x for x, _ in qr_01]).mean().round(3)
np.array([np.abs(x) for _, x in qr_01]).mean().round(3)