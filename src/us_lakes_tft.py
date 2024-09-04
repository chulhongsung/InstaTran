#%%
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
from models import TemporalFusionTransformer 

from tqdm import tqdm
import pandas as pd
import numpy as np 

from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import argparse

parser = argparse.ArgumentParser(description='hyperparams')
parser.add_argument('--epochs', required=False, default=1000, type=int)
parser.add_argument('--bs', required=False, default=256, type=int)
parser.add_argument('--lr', required=False, default=0.001, type=float)
parser.add_argument('--seed', required=False, default=42, type=int)
parser.add_argument('--d_emb', required=False, default=1, type=int)
parser.add_argument('--d_model', required=False, default=6, type=int)
parser.add_argument('--dr', required=False, default=0.1, type=float)

args = parser.parse_args()

from datetime import datetime
now = datetime.now().strftime("%y-%m%d-%H%M")

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

def train(model, loader, criterion, optimizer, device):
    
    model.train()
    
    qr_loss = []
    
    for batch in loader:
        conti_input, cate_input, future_input, true_y = batch 
        
        conti_input = conti_input.to(device)
        cate_input = cate_input.to(device)
        future_input = future_input.to(device)
        true_y = true_y.to(device)
        
        pred = model(conti_input, cate_input, future_input)
        
        loss = criterion(true_y, pred.squeeze())
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        qr_loss.append(loss)
        
    return sum(qr_loss)/len(qr_loss)

class QuantileRisk(nn.Module):
    def __init__(self, tau, quantile, device):
        super(QuantileRisk, self).__init__()
        self.quantile = quantile
        self.device = device
        self.q_arr = torch.tensor(quantile).float().unsqueeze(-1).repeat(1, 1, tau).transpose(-1, -2).to(self.device)
    
    def forward(self, true, pred):
        
        ql = torch.maximum(self.q_arr * (true.unsqueeze(-1) - pred), (1-self.q_arr)*(pred - true.unsqueeze(-1)))

        return ql.mean() * 1000

def main():

    config = {
        "epochs": args.epochs,
        "lr": args.lr,
        'bs': args.bs,
        'd_model': args.d_model,
        'd_emb': args.d_emb,
        'dr': args.dr,
    }

    data_split_range = [(2005, 2013), (2008, 2016), (2011, 2019), (2014, 2022)]

    ql_09 = []
    ql_05 = []
    ql_01 = []

    qr_09 = []
    qr_05 = []
    qr_01 = []
    
    torch.manual_seed(args.seed)
    
    for a, b in data_split_range:
        
        tmp_best_loss = torch.inf
        
        tmp_train, tmp_valid, tmp_test, scaler = train_valid_test_split_for_dl(df_mead.loc[(df_mead["year"] >= a) & (df_mead["year"] <= b)],
                                                df_mohave.loc[(df_mohave["year"] >= a) & (df_mohave["year"] <= b)],
                                                df_havasu.loc[(df_havasu["year"] >= a) & (df_havasu["year"] <= b)])
        
        train_dataset = TensorDataset(torch.FloatTensor(tmp_train[0]), torch.LongTensor(tmp_train[1]), torch.LongTensor(tmp_train[2]), torch.FloatTensor(tmp_train[3]))
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config["bs"])

        valid_input1 = torch.FloatTensor(tmp_valid[0])
        valid_input2 = torch.LongTensor(tmp_valid[1])
        valid_input3 = torch.LongTensor(tmp_valid[2])
        valid_label = torch.FloatTensor(tmp_valid[3])

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        model = TemporalFusionTransformer(
            d_model=config["d_model"],
            d_embedding=config["d_emb"],
            cate_dims=[13, 32],
            num_cv=11,
            seq_len=24,
            num_targets=1,
            tau=4,
            quantile=[0.1, 0.5, 0.9],
            dr=config["dr"],
            device=device
        )
        
        model.to(device)
        criterion = QuantileRisk(4, [0.1, 0.5, 0.9], device)
        optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
        
        # pbar = range(config["epochs"])

        for epoch in range(config["epochs"]):        
            train_loss = train(model, train_loader, criterion, optimizer, device)
            # pbar.set_description("Train Loss: {:.4f}".format(train_loss))
            
            model.eval()
            
            if (epoch >= 200) & (epoch % 50 == 0):      
                with torch.no_grad():        
                    valid_pred = model(valid_input1, valid_input2, valid_input3)
                    tmp_valid_loss = criterion(valid_label, valid_pred.squeeze())
                    
                    if tmp_best_loss > tmp_valid_loss:
                        # print("Valid Best Update!")
                        torch.save(model.state_dict(), f'../assets/weights/us_lakes_tft_valid_best_{a}_{b}.pth')
                        tmp_best_loss = tmp_valid_loss
                        
        torch.save(model.state_dict(), f'../assets/weights/us_lakes_tft_final_{a}_{b}.pth')
        
        valid_best_model = TemporalFusionTransformer(
            d_model=config["d_model"],
            d_embedding=config["d_emb"],
            cate_dims=[13, 32],
            num_cv=11,
            seq_len=24,
            num_targets=1,
            tau=4,
            quantile=[0.1, 0.3, 0.5, 0.7, 0.9],
            dr=config["dr"],
            device=device
        )
        
        valid_best_model.load_state_dict(torch.load(f'../assets/weights/us_lakes_tft_valid_best_{a}_{b}.pth', map_location='cpu'))
        
        test_input1 = torch.FloatTensor(tmp_test[0])
        test_input2 = torch.LongTensor(tmp_test[1])
        test_input3 = torch.LongTensor(tmp_test[2])
        label = tmp_test[3]
        
        valid_best_model.eval()    
        with torch.no_grad():
            pred = valid_best_model(test_input1, test_input2, test_input3)
            pred = pred.squeeze().detach().cpu().numpy()

        ql_09.append(np.maximum(0.9 * (label - pred[..., 2]), (1-0.9)*(pred[..., 2] - label)).mean() * 1000)
        ql_05.append(np.maximum(0.5 * (label - pred[..., 1]), (1-0.5)*(pred[..., 1] - label)).mean() * 1000)
        ql_01.append(np.maximum(0.1 * (label - pred[..., 0]), (1-0.1)*(pred[..., 0] - label)).mean() * 1000)
            
        qr_09.append((np.mean(label < pred[..., 2]), 0.9 - np.mean(label < pred[..., 2])))
        qr_05.append((np.mean(label < pred[..., 1]), 0.5 - np.mean(label < pred[..., 1])))
        qr_01.append((np.mean(label < pred[..., 0]), 0.1 - np.mean(label < pred[..., 0])))
    
    np.array([x for x, _ in qr_09]).mean().round(3)
    np.array([np.abs(x) for _, x in qr_09]).mean().round(3)

    np.array([x for x, _ in qr_05]).mean().round(3)
    np.array([np.abs(x) for _, x in qr_05]).mean().round(3)

    np.array([x for x, _ in qr_01]).mean().round(3)
    np.array([np.abs(x) for _, x in qr_01]).mean().round(3)
    
if __name__ == '__main__':
    main()