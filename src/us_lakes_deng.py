import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset #IterableDataset
from torch.nn.utils import weight_norm

import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from utils import *
from layers import *
from models import STALSTM

from tqdm import tqdm
import pandas as pd
import numpy as np 

from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

os.chdir(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
df_mead = pd.read_csv("../data/us_lakes/df_mead_preprocessed.csv")
df_mohave = pd.read_csv("../data/us_lakes/df_mohave_preprocessed.csv")
df_havasu = pd.read_csv("../data/us_lakes/df_havasu_preprocessed.csv")

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size  

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=4, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i 
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class GTCN(nn.Module):
    def __init__(self, input_size, seq_len, num_channels, kernel_size=8, dropout=0.1):
        super(GTCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear1 = nn.Linear(seq_len, seq_len)
        self.linear2 = nn.Linear(seq_len, seq_len)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.tcn(x)
        x1 = self.linear1(x)
        x2 = self.sigmoid(self.linear2(x))
        
        return (x1 * x2).unsqueeze(-2).transpose(3, 1).transpose(3, 2)

def nconv(x, A):
    return torch.einsum('ncvl,vw->ncwl', (x, A)).contiguous()

class GraphConvNet(nn.Module):
    def __init__(self, adj, node_dim, dropout):
        super(GraphConvNet, self).__init__()
        self.dropout = dropout
        self.linears = nn.ModuleList([nn.Linear(node_dim, node_dim) for i in range(2)])
        self.adj = adj 
    def forward(self, x):
        for _,l in enumerate(self.linears):
            x = nconv(l(x), self.adj)
        h = F.dropout(x, self.dropout, training=self.training)
        return h

class GraphAttentionNet(nn.Module):
    def __init__(self, adj, in_feature, out_feature, dropout, concat=True):
        super(GraphAttentionNet, self).__init__()
        self.dropout = dropout
        self.in_features = in_feature
        self.out_features = out_feature
        self.concat = concat
        self.adj = adj
        self.W = nn.Parameter(torch.empty(size=(self.in_features, self.out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*self.out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU()

    def forward(self, h):
        Wh = torch.matmul(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(self.adj > 0, e, zero_vec)
        alpha_ = F.softmax(attention, dim=-1)
        alpha = F.dropout(alpha_, self.dropout, training=self.training)
        h_prime = torch.matmul(alpha, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.transpose(3, 2)
        return self.leakyrelu(e)

class GraphFusion(nn.Module):
    def __init__(self, adj, node_dim, in_feature, out_feature, dropout):
        super(GraphFusion, self).__init__()
        self.gcn = GraphConvNet(adj=adj, node_dim=node_dim, dropout=dropout)
        self.gat = GraphAttentionNet(adj=adj, in_feature=in_feature, out_feature=out_feature, dropout=dropout)
        self.adj = adj
    def forward(self, x):
        x1 = self.gcn(x)
        x2 = self.gat(x)
        
        return x1 + x2
        
class HSDSTM(nn.Module):
    """Deng et al, 2023 (Stochastic Environmental Research and Risk Assessment)

    Args:
        nn (_type_): _description_
    """
    def __init__(self, 
                 adj = None,
                 input_size=11,
                 seq_len=24,
                 num_channels=[11, 11],
                 node_dim=1,
                 dropout=0.1,
                 num_levels=2,
                 tau=4,
                 num_quantiles=5
                 ):
        super(HSDSTM, self).__init__()
        self.node_dim = node_dim
        self.dropout = dropout
        self.adj = adj
        
        self.levels = nn.ModuleList([nn.Sequential(
            GTCN(input_size=input_size, seq_len=seq_len, num_channels=num_channels),
            GraphFusion(adj=self.adj, node_dim=node_dim, in_feature=node_dim, out_feature=node_dim, dropout=dropout)
        ) for _ in range(num_levels)])
        
        self.qol = nn.ModuleList([nn.Linear(seq_len, tau) for _ in range(num_quantiles)])
        
    def forward(self, x):
        output_list = []
        
        for _, l in enumerate(self.levels):
            h = l(x)
            h = h + x.transpose(2, 1).unsqueeze(-1)
            output_list.append(h)
            x = x.squeeze()
        
        fusion = torch.cat(output_list, dim=-1).mean(dim=-2).mean(dim=-1)
        
        total_output_list = []
        
        for _,l in enumerate(self.qol):
            tmp_quantile_output = l(fusion)
            total_output_list.append(tmp_quantile_output.unsqueeze(-1))
        
        return torch.cat(total_output_list, dim=-1)
    
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
        
        pred = model(conti_input)
        
        loss = criterion(true_y, pred)
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        total_loss.append(loss)
        
    return sum(total_loss)/len(total_loss)

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

data_split_range = [(2005, 2013), (2008, 2016), (2011, 2019), (2014, 2022)]

adj = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], 
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
                ).float()

norm_adj = adj/adj.sum(dim=-1).unsqueeze(-1)

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
     
    train_dataset = TensorDataset(torch.FloatTensor(np.swapaxes(tmp_train[0], 2, 1)), torch.FloatTensor(tmp_train[2]))
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=256)

    deng = HSDSTM(adj=norm_adj,
                 input_size=11,
                 seq_len=24,
                 num_channels=[11, 11],
                 node_dim=1,
                 dropout=0.05,
                 num_levels=2,
                 tau=4,
                 num_quantiles=5)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    deng.to(device)
    optimizer = optim.AdamW(deng.parameters(), lr=0.001)
    criterion = QuantileRisk(4, [0.1, 0.3, 0.5, 0.7, 0.9], device)
    
    pbar = tqdm(range(1500))

    for epoch in pbar:        
        train_loss = train(deng, train_loader, criterion, optimizer, device)
        pbar.set_description("Train Loss: {:.4f}".format(train_loss))
    
    torch.save(deng.state_dict(), f'../assets/weights/deng_us_lakes_{a}_{b}.pth')
    
    test_input = torch.FloatTensor(np.swapaxes(tmp_test[0], 2, 1))
    label = tmp_test[2]
    
    deng.eval()    
    with torch.no_grad():
        pred_results = deng(test_input)
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