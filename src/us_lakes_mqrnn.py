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

from tqdm import tqdm
import pandas as pd
import numpy as np 

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
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
    train_future = np.zeros((n, tau, 2))
        
    for j in range(n):
        train_input[j, :] = tmp_arr[j:(j+input_seq_len)]
        train_label[j, :] = tmp_label[(j+input_seq_len):(j+input_seq_len+tau)]/1000
        train_date[j, :] = tmp_date[j:(j+input_seq_len)]
        train_future[j, :] = tmp_date[(j+input_seq_len):(j+input_seq_len+tau)]
        
    tmp_arr = np.array(df_valid)
    tmp_label = np.array(df_valid[:, col_labels])
    tmp_date = np.array(df_valid_date)
    
    _, p = tmp_arr.shape
    n = tmp_arr.shape[0] - input_seq_len - tau 
    
    valid_input = np.zeros((n, input_seq_len, p), dtype=np.float32)
    valid_label = np.zeros((n, tau))
    valid_date = np.zeros((n, input_seq_len, 2))
    valid_future = np.zeros((n, tau, 2))
    
    for j in range(n):
        valid_input[j, :] = tmp_arr[j:(j+input_seq_len)]
        valid_label[j, :] = tmp_label[(j+input_seq_len):(j+input_seq_len+tau)]/1000
        valid_date[j, :] = tmp_date[j:(j+input_seq_len)]
        valid_future[j, :] = tmp_date[(j+input_seq_len):(j+input_seq_len+tau)]
        
    tmp_arr = np.array(df_test)
    tmp_label = np.array(df_test[:, col_labels])
    tmp_date = np.array(df_test_date)
    
    _, p = tmp_arr.shape
    n = tmp_arr.shape[0] - input_seq_len - tau 
    
    test_input = np.zeros((n, input_seq_len, p), dtype=np.float32)
    test_label = np.zeros((n, tau))
    test_date = np.zeros((n, input_seq_len, 2))
    test_future = np.zeros((n, tau, 2))
    
    for j in range(n):
        test_input[j, :] = tmp_arr[j:(j+input_seq_len)]
        test_label[j, :] = tmp_label[(j+input_seq_len):(j+input_seq_len+tau)]/1000
        test_date[j, :] = tmp_date[j:(j+input_seq_len)]
        test_future[j, :] = tmp_date[(j+input_seq_len):(j+input_seq_len+tau)]
    
    scaler.fit(train_input.reshape(train_input.shape[0], -1))
    train_scaled = scaler.transform(train_input.reshape(train_input.shape[0], -1)).reshape(train_input.shape[0], input_seq_len, -1)
    valid_scaled = scaler.transform(valid_input.reshape(valid_input.shape[0], -1)).reshape(valid_input.shape[0], input_seq_len, -1)
    test_scaled = scaler.transform(test_input.reshape(test_input.shape[0], -1)).reshape(test_input.shape[0], input_seq_len, -1)
    
    return (train_scaled, train_date, train_future, train_label), (valid_scaled, valid_date, valid_future, valid_label), (test_scaled, test_date, test_future, test_label), scaler

# %%
class Encoder(nn.Module):
    def __init__(self, d_input, d_embedding, n_embedding, d_model, n_layers=3, dr=0.1):
        super(Encoder, self).__init__()
        self.embedding_layers = nn.ModuleList([nn.Embedding(n, d_embedding) for n in n_embedding]) 
        self.lstm = nn.LSTM(d_input + len(n_embedding) * d_embedding, d_model, n_layers, dropout=dr, batch_first=True)
        
    def forward(self, conti, cate):
        tmp_feature_list = []
        
        for i, l in enumerate(self.embedding_layers):
            tmp_feature = l(cate[:, :, i:i+1])
            tmp_feature_list.append(tmp_feature)
            
        emb_output = torch.cat(tmp_feature_list, axis=-2)
        emb_output = emb_output.view(conti.size(0), conti.size(1), -1)
        
        x = torch.cat([conti, emb_output], axis=-1)
        
        _, (hidden, cell) = self.lstm(x)

        return hidden, cell

class GlobalDecoder(nn.Module):
    def __init__(self, d_hidden:int, d_embedding:int, n_embedding:list, d_model:int, tau:int, num_targets:int, dr:float):
        super(GlobalDecoder, self).__init__()
        self.d_hidden = d_hidden
        self.d_embedding = d_embedding
        self.n_embedding = n_embedding
        self.d_model = d_model
        self.tau = tau
        self.num_targets = num_targets
        self.dr = dr
        self.embedding_layers = nn.ModuleList([nn.Embedding(n, d_embedding) for n in n_embedding]) 
        self.linear_layers = nn.ModuleList([nn.Linear(d_hidden + tau * d_embedding * len(n_embedding), (tau+1) * d_model) for _ in range(num_targets)])
        self.dropout = nn.Dropout(dr)
        
    def forward(self, future, hidden):
        tmp_feature_list = []
        
        for i, l in enumerate(self.embedding_layers):
            tmp_feature = l(future[:, :, i:i+1])
            tmp_feature_list.append(tmp_feature)
            
        emb_output_ = torch.cat(tmp_feature_list, axis=-2)
        emb_output = emb_output_.view(future.size(0), -1)
        
        num_layers, batch_size, d_hidden = hidden.size()
        
        assert d_hidden == self.d_model 
        
        x = torch.cat([hidden[num_layers-1], emb_output], axis=-1)
        
        tmp_global_context = []
        for l in self.linear_layers:
            tmp_gc = self.dropout(l(x))
            tmp_global_context.append(tmp_gc.unsqueeze(1))
        
        global_context = torch.cat(tmp_global_context, axis=1)
        
        return emb_output_.view(batch_size, self.tau, -1), global_context # (batch_size, tau, d_embedding * len(n_embedding)), (batch_size, num_targets, (tau+1) * d_model), (tau+1): c_{a} , c_{t+1:t+tau}

class LocalDecoder(nn.Module):
    def __init__(self, d_hidden:int, d_embedding:int, n_embedding: list, d_model:int, tau:int, num_targets:int, num_quantiles:int, dr:float):
        super(LocalDecoder, self).__init__()
        self.d_hidden = d_hidden
        self.d_embedding = d_embedding
        self.n_embedding = n_embedding
        self.d_model = d_model
        self.tau = tau
        self.num_targets = num_targets
        self.dr = dr
        self.linear_layers = nn.Sequential(
            nn.Linear(2 * d_model + d_embedding * len(n_embedding), d_model * 2),
            nn.Dropout(dr),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dr),
            nn.Linear(d_model, num_quantiles)            
            )
                
    def forward(self, embedded_future, global_output):
        batch_size = global_output.size(0)
        
        c_a = global_output[..., :self.d_model].unsqueeze(-2).repeat(1, 1, self.tau, 1) # (batch_size, num_targets, tau, d_model)
        c_t = global_output[..., self.d_model:].view(batch_size, self.num_targets, self.tau, -1) # (batch_size, num_targets, tau, d_model)
        x_ = torch.cat([c_a,c_t.view(batch_size, self.num_targets, self.tau, -1)], axis=-1) # (batch_size, num_targets, tau, 2*d_model)
        x = torch.cat([x_, embedded_future.unsqueeze(1).repeat(1, self.num_targets, 1, 1)], axis=-1) # (batch_size, num_targets, tau, 2*d_model + d_embedding * len(n_embedding))
        
        output = self.linear_layers(x)
        
        return output.transpose(2, 1).squeeze() # (batch_size, tau, num_targets, num_quantiles)
    
class MQRnn(nn.Module):
    def __init__(self, d_input:int, d_embedding:int, n_embedding:list, d_model:int, tau:int, num_targets:int, num_quantiles: int, n_layers:int, dr:float):
        super(MQRnn, self).__init__()
        self.encoder = Encoder(
                               d_input=d_input,
                               d_embedding=d_embedding,
                               n_embedding=n_embedding,
                               d_model=d_model,
                               n_layers=n_layers,
                               dr=dr
                               )
        self.global_decoder = GlobalDecoder(
                                            d_hidden=d_model,
                                            d_embedding=d_embedding,
                                            n_embedding=n_embedding,
                                            d_model=d_model,
                                            tau=tau,
                                            num_targets=num_targets,
                                            dr=dr
                                            )
        self.local_decoder = LocalDecoder(
                                          d_hidden=d_model,
                                          d_embedding=d_embedding,
                                          n_embedding=n_embedding,
                                          d_model=d_model,
                                          tau=tau,
                                          num_targets=num_targets,
                                          num_quantiles=num_quantiles,
                                          dr=dr
                                          )
        
    def forward(self, conti, cate, future):
        hidden, _ = self.encoder(conti, cate)
        embedded_future, global_output = self.global_decoder(future, hidden)
        output = self.local_decoder(embedded_future, global_output)
        
        return output

def train(model, loader, criterion, optimizer, device):
    
    model.train()
    
    total_loss = []
    
    for batch in loader:
        conti_input, cate_input, future_input, true_y = batch 
        
        conti_input = conti_input.to(device)
        cate_input = cate_input.to(device)
        future_input = future_input.to(device)
        true_y = true_y.to(device)
        
        pred = model(conti_input, cate_input, future_input)
        
        loss = criterion(true_y, pred)
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        total_loss.append(loss)
        
    return sum(total_loss)/len(total_loss)
#%%
class QuantileRisk(nn.Module):
    def __init__(self, tau, quantile, device):
        super(QuantileRisk, self).__init__()
        self.quantile = quantile
        self.device = device
        self.q_arr = torch.tensor(quantile).float().unsqueeze(-1).repeat(1, 1, tau).transpose(-1, -2).to(self.device)
    
    def forward(self, true, pred):
        
        ql = torch.maximum(self.q_arr * (true.unsqueeze(-1) - pred), (1-self.q_arr)*(pred - true.unsqueeze(-1)))

        return ql.mean() * 1000
#%%
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
     
    train_dataset = TensorDataset(torch.FloatTensor(tmp_train[0]), torch.LongTensor(tmp_train[1]), torch.LongTensor(tmp_train[2]), torch.FloatTensor(tmp_train[3]))
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=256)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MQRnn(
        d_input=11, 
        d_embedding=1, 
        n_embedding=[13, 32], 
        d_model=20, 
        tau=4,
        num_targets=1,
        num_quantiles=5,
        n_layers=3,
        dr=0.05
    )
        
    
    model.to(device)
    criterion = QuantileRisk(4, [0.1, 0.3, 0.5, 0.7, 0.9], device)
    optimizer = optim.AdamW(model.parameters(), lr=0.003)
    
    pbar = tqdm(range(600))

    for epoch in pbar:        
        train_loss = train(model, train_loader, criterion, optimizer, device)
        pbar.set_description("Train Loss: {:.4f}".format(train_loss))
        
    test_input1 = torch.FloatTensor(tmp_test[0])
    test_input2 = torch.LongTensor(tmp_test[1])
    test_input3 = torch.LongTensor(tmp_test[2])
    label = tmp_test[3]
    
    model.eval()    
    with torch.no_grad():
        pred = model(test_input1, test_input2, test_input3)
        pred = pred.detach().cpu().numpy()

    ql_09.append(np.maximum(0.9 * (label - pred[..., 4]), (1-0.9)*(pred[..., 4] - label)).mean() * 1000)
    ql_07.append(np.maximum(0.7 * (label - pred[..., 3]), (1-0.7)*(pred[..., 3] - label)).mean() * 1000)
    ql_05.append(np.maximum(0.5 * (label - pred[..., 2]), (1-0.5)*(pred[..., 2] - label)).mean() * 1000)
    ql_03.append(np.maximum(0.3 * (label - pred[..., 1]), (1-0.3)*(pred[..., 1] - label)).mean() * 1000)
    ql_01.append(np.maximum(0.1 * (label - pred[..., 0]), (1-0.1)*(pred[..., 0] - label)).mean() * 1000)
        
    qr_09.append((np.mean(label < pred[..., 4]), 0.9 - np.mean(label < pred[..., 4])))
    qr_07.append((np.mean(label < pred[..., 3]), 0.7 - np.mean(label < pred[..., 3])))
    qr_05.append((np.mean(label < pred[..., 2]), 0.5 - np.mean(label < pred[..., 2])))
    qr_03.append((np.mean(label < pred[..., 1]), 0.3 - np.mean(label < pred[..., 1])))
    qr_01.append((np.mean(label < pred[..., 0]), 0.1 - np.mean(label < pred[..., 0])))