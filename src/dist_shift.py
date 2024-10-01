# Experiment of distribution shift 
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
from models import InstaTran

import argparse

parser = argparse.ArgumentParser(description='hyperparams')
parser.add_argument('--epochs', required=False, default=300, type=int)
parser.add_argument('--bs', required=False, default=200, type=int)
parser.add_argument('--lr', required=False, default=0.001, type=float)
parser.add_argument('--seed', required=False, default=1, type=int)
parser.add_argument('--d_emb', required=False, default=3, type=int)
parser.add_argument('--d_model', required=False, default=20, type=int)
parser.add_argument('--dr', required=False, default=0.1, type=float)
parser.add_argument('--tau', required=False, default=12, type=int)
parser.add_argument('--seq_len', required=False, default=48, type=int)
parser.add_argument('--year', required=False, default="2016", choices=("2016", "2017", "2018", "2019", "2020", "2021"), type=str)
args = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model, loader, criterion, optimizer, device):
    
    model.train()
    
    qr_loss = []
    
    for batch in loader:
        conti_input, cate_input, future_input, true_y = batch 
        
        conti_input = conti_input.to(device)
        cate_input = cate_input.to(device)
        future_input = future_input.to(device)
        true_y = true_y.to(device)
        
        pred, _, _, _, _, _, _ = model(conti_input, cate_input, future_input)
        
        loss = criterion(true_y, pred)
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        qr_loss.append(loss)
        
    return sum(qr_loss)/len(qr_loss)
    
def evaluate(model, loader, criterion, device):
    model.eval()
    
    qr_loss = []
    
    for batch in loader:
        conti_input, cate_input, future_input, true_y = batch 
        
        conti_input = conti_input.to(device)
        cate_input = cate_input.to(device)
        future_input = future_input.to(device)
        true_y = true_y.to(device)
        
        pred, ssa_weight1, ssa_weight2, tsa_weight, dec_weights, _, _ = model(conti_input, cate_input, future_input)
        
        loss = criterion(true_y, pred)
        
        qr_loss.append(loss)
        
        return sum(qr_loss)/len(qr_loss), ssa_weight1[:1, :1, ...], ssa_weight2[:1, :1, ...], tsa_weight[:1, ...], dec_weights[:1, ...]
    
def main():

    df_train_total = pd.read_csv("../data/df_train_total_ds_{}.csv".format(args.year))
    df_test_total = pd.read_csv("../data/df_test_total_ds_{}.csv".format(args.year))
    df_merged = pd.read_csv("../data/df_merged_ds_{}.csv".format(args.year))
    
    train_conti_input, train_cate_input, train_future_input, train_label = generate_ts_data(df_train_total, df_merged, input_seq_len=args.seq_len, tau=args.tau)
    test_conti_input, test_cate_input, test_future_input, test_label = generate_ts_data(df_test_total, df_merged, input_seq_len=args.seq_len, tau=args.tau)

    torch.manual_seed(args.seed)
    
    num_targets = 1
    
    from torch.utils.data import DataLoader, TensorDataset #IterableDataset
        
    train_dataset = TensorDataset(torch.FloatTensor(train_conti_input), torch.LongTensor(train_cate_input), torch.LongTensor(train_future_input), torch.FloatTensor(train_label))
    test_dataset = TensorDataset(torch.FloatTensor(test_conti_input), torch.LongTensor(test_cate_input), torch.LongTensor(test_future_input), torch.FloatTensor(test_label))
    
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.bs)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=args.bs)
    
    quanilte_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    qr = QuantileRisk(args.tau, quanilte_levels, num_targets, device)
    

    sps = torch.tensor([ [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
                    ).float()

    
    instatran = InstaTran(
        d_model=args.d_model,
        d_embedding=args.d_emb,
        cate_dims=[16, 32, 24],
        spatial_structure=sps,
        num_cv=16,
        seq_len=args.seq_len,
        num_targets=num_targets,
        tau=args.tau,
        quantile=quanilte_levels,
        dr=args.dr,
        device=device
    ) 
        
    instatran.to(device)
        
    optimizer = optim.Adam(instatran.parameters(), lr=args.lr)
    
    tmp_val_loss = np.infty
       
    for epoch in range(args.epochs):        

        train_loss = train(instatran, train_loader, qr, optimizer, device)
        
        if epoch % 20 == 0:
            
            eval_loss, _, ssa_weight2, tsa_weight, dec_weights = evaluate(instatran, test_loader, qr, device)
    
            
            if eval_loss.cpu().item() < tmp_val_loss:
                tmp_val_loss = eval_loss.cpu().item()
                best_eval_model = instatran
                   
    # torch.save(instatran.state_dict(), '../assets/ds/ds_InstaTran_{}_final.pth'.format(args.year))        
    # torch.save(best_eval_model.state_dict(), '../assets/ds/ds_InstaTran_{}_best.pth'.format(args.year))
    
if __name__ == '__main__':
    main()
