# Experiment of distribution shift 
import torch
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd

import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from utils import *
from layers import *
from models import STALSTM, HSDSTM
    
import argparse

parser = argparse.ArgumentParser(description='hyperparams')
parser.add_argument('--epochs', required=False, default=300, type=int)
parser.add_argument('--bs', required=False, default=200, type=int)
parser.add_argument('--lr', required=False, default=0.001, type=float)
parser.add_argument('--seed', required=False, default=1, type=int)
parser.add_argument('--dr', required=False, default=0.1, type=float)
parser.add_argument('--tau', required=False, default=12, type=int)
parser.add_argument('--seq_len', required=False, default=48, type=int)
parser.add_argument('--model', required=False, default="Deng", choices=("STALSTM", "HSDSTM"), type=str)
parser.add_argument('--year', required=False, default="2016", choices=("2016", "2017", "2018", "2019", "2020", "2021"), type=str)

args = parser.parse_args()
    
def main():
        
    df_train_total = pd.read_csv("../data/df_train_total_ds_{}.csv".format(args.year))
    df_test_total = pd.read_csv("../data/df_test_total_ds_{}.csv".format(args.year))
    df_merged = pd.read_csv("../data/df_merged_ds_{}.csv".format(args.year))
    
    torch.manual_seed(args.seed)
    quanilte_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
     
    config = {
        'seed': args.seed,
        'bs': args.bs,
        'seq_len': args.seq_len,
        'tau': args.tau,
        'epochs': args.epochs,
        'lr': args.lr,
        'dr': args.dr
    }
    
    from torch.utils.data import DataLoader, TensorDataset 

    if args.model == "STALSTM":
        model = STALSTM(48, 16, 12, 5)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = QuantileRisk(args.tau, quanilte_levels, 1, device)
    
        def generate_ts_data(df, label_df, input_seq_len=48, tau=12):
            conti_input_list = []
            cate_input_list = []
            future_input_list = []
            label_list = []
            col_labels =  ['wl_1018680'] 
            
            for i in df['year'].unique():
                tmp_df = np.array(df.loc[df['year'] == i, :])
                tmp_label_df = np.array(label_df.loc[label_df['year'] == i, col_labels])
                n = tmp_df.shape[0] - input_seq_len - tau 
                
                tmp_conti_input = tmp_df[:, 4:] 
                tmp_cate_input = tmp_df[:, 1:4] 
                
                conti_input = np.zeros((n, input_seq_len, tmp_conti_input.shape[1]), dtype=np.float32)
                cate_input = np.zeros((n, input_seq_len, tmp_cate_input.shape[1]), dtype=np.float32)
                future_input = np.zeros((n, tau, tmp_cate_input.shape[1]), dtype=np.float32)
                label = np.zeros((n, tau, len(col_labels)))
            
                for j in range(n):
                    conti_input[j, :, :] = tmp_conti_input[j:(j+input_seq_len), :]
                    cate_input[j, :, :] = tmp_cate_input[j:(j+input_seq_len), :]
                    future_input[j, :, :] = tmp_cate_input[(j+input_seq_len):(j+input_seq_len+tau), :]
                    label[j, :, :] = tmp_label_df[(j+input_seq_len):(j+input_seq_len+tau), :]/1000

                conti_input_list.append(conti_input)
                cate_input_list.append(cate_input)
                future_input_list.append(future_input)
                label_list.append(label)
            
            total_conti_input = np.concatenate(conti_input_list, axis=0)
            total_cate_input = np.concatenate(cate_input_list, axis=0)
            total_future_input = np.concatenate(future_input_list, axis=0)
            total_label = np.concatenate(label_list, axis=0)
            
            return total_conti_input, total_cate_input, total_future_input, total_label

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
            
        def evaluate(model, loader, criterion, device):
            model.eval()
            
            qr_loss = []
            
            for batch in loader:
                conti_input, true_y = batch 
                
                conti_input = conti_input.to(device)
                true_y = true_y.to(device)
                
                pred, _, _ = model(conti_input)
                
                loss = criterion(true_y, pred)
                
                qr_loss.append(loss)
                
                return sum(qr_loss)/len(qr_loss)
            
        train_conti_input, _, _, train_label = generate_ts_data(df_train_total, df_merged, input_seq_len=args.seq_len, tau=args.tau)
        test_conti_input, _, _, test_label = generate_ts_data(df_test_total, df_merged, input_seq_len=args.seq_len, tau=args.tau)
        
        train_dataset = TensorDataset(torch.FloatTensor(train_conti_input), torch.FloatTensor(train_label))
        test_dataset = TensorDataset(torch.FloatTensor(test_conti_input), torch.FloatTensor(test_label))

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.bs)
        test_loader = DataLoader(test_dataset, shuffle=True, batch_size=args.bs)

    elif args.model == "HSDSTM":
        adj = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                            [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                            [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]]
                ).float()

        norm_adj = adj/adj.sum(dim=-1).unsqueeze(-1)
        norm_adj = norm_adj.to(device)
        model = HSDSTM(adj=norm_adj,                  
                       input_size=16,
                       seq_len=48,
                       num_channels=[16, 16],
                       node_dim=1,
                       dropout=0.1,
                       num_levels=3,
                       tau=12,
                       num_quantiles=5)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = QuantileRisk(12, [0.1, 0.3, 0.5, 0.7, 0.9], 1, device)
        
        def generate_ts_data(df, label_df, input_seq_len=48, tau=12):
            conti_input_list = []
            label_list = []
            col_labels =  ['wl_1018680'] # ['wl_1018662', 'wl_1018680', 'wl_1018683', 'wl_1019630']
            
            for i in df['year'].unique():
                tmp_df = np.array(df.loc[df['year'] == i, :])
                tmp_label_df = np.array(label_df.loc[label_df['year'] == i, col_labels])
                n = tmp_df.shape[0] - input_seq_len - tau 
                
                tmp_conti_input = tmp_df[:, 4:] # (4416, 16)
                
                conti_input = np.zeros((n, input_seq_len, tmp_conti_input.shape[1]), dtype=np.float32)
                label = np.zeros((n, tau, len(col_labels)))
            
                for j in range(n):
                    conti_input[j, :, :] = tmp_conti_input[j:(j+input_seq_len), :]
                    label[j, :, :] = tmp_label_df[(j+input_seq_len):(j+input_seq_len+tau), :]/1000

                conti_input_list.append(conti_input)
                label_list.append(label)
            
            total_conti_input = np.concatenate(conti_input_list, axis=0)
            total_label = np.concatenate(label_list, axis=0)
            
            return np.swapaxes(total_conti_input, 2, 1), total_label   
        
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
        
        def evaluate(model, loader, criterion, device):
            model.eval()
            
            qr_loss = []
            
            for batch in loader:
                conti_input, true_y = batch 
                
                conti_input = conti_input.to(device)
                true_y = true_y.to(device)
                
                pred = model(conti_input)
                
                loss = criterion(true_y, pred)
                
                qr_loss.append(loss)
                
                return sum(qr_loss)/len(qr_loss)
        
        train_conti_input, train_label = generate_ts_data(df_train_total, df_merged)
        train_dataset = TensorDataset(torch.FloatTensor(train_conti_input), torch.FloatTensor(train_label))
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.bs)
        
        test_conti_input, test_label = generate_ts_data(df_test_total, df_merged)
        test_dataset = TensorDataset(torch.FloatTensor(test_conti_input), torch.FloatTensor(test_label))
        test_loader = DataLoader(test_dataset, shuffle=True, batch_size=args.bs)
    
    tmp_val_loss = np.infty
       
    for epoch in range(args.epochs):        

        train_loss = train(model, train_loader, criterion, optimizer, device)
        
        if epoch % 20 == 0:
            
            eval_loss  = evaluate(model, test_loader, criterion, device)

            if eval_loss.cpu().item() < tmp_val_loss:
                tmp_val_loss = eval_loss.cpu().item()
                best_eval_model = model
       
    # torch.save(model.state_dict(), '../assets/ds/ds_{}_{}_final.pth'.format(args.model, args.year))
    # torch.save(best_eval_model.state_dict(), '../assets/ds/ds_{}_{}_best.pth'.format(args.model, args.year))

    
if __name__ == '__main__':
    main()
