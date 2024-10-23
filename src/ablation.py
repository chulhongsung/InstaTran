import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset 

import numpy as np
import pandas as pd

import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from utils import *
from layers import *
from models import *

import argparse

parser = argparse.ArgumentParser(description='hyperparams')
parser.add_argument('--epochs', required=False, default=200, type=int)
parser.add_argument('--bs', required=False, default=500, type=int)
parser.add_argument('--lr', required=False, default=0.001, type=float)
parser.add_argument('--seed', required=False, default=7, type=int)
parser.add_argument('--tau', required=False, default=12, type=int)
parser.add_argument('--seq_len', required=False, default=48, type=int)

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
    
    df_train_total = pd.read_csv("../data/df_train_total.csv")
    df_merged = pd.read_csv("../data/df_merged.csv")
    
    train_conti_input, train_cate_input, train_future_input, train_label = generate_ts_data(df_train_total, df_merged, input_seq_len=args.seq_len, tau=args.tau)

    torch.manual_seed(args.seed)
    
    num_targets = 1
    
    train_dataset = TensorDataset(torch.FloatTensor(train_conti_input), torch.LongTensor(train_cate_input), torch.LongTensor(train_future_input), torch.FloatTensor(train_label))
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.bs)
    
    quantile_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    
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


    instatran_wo_sps = SpatialTemporalTransformer(
        d_model=30,
        d_embedding=5,
        cate_dims=[16, 32, 24],
        spatial_structure=None,
        num_cv=16,
        seq_len=48,
        num_targets=1,
        tau=12,
        quantile=[0.1, 0.3, 0.5, 0.7, 0.9],
        dr=0.1,
        device=device
    )

    instatran_parallel = SpatialTemporalParallelTransformer(
        d_model=30,
        d_embedding=5,
        cate_dims=[16, 32, 24],
        spatial_structure=None,
        num_cv=16,
        seq_len=48,
        num_targets=1,
        tau=12,
        quantile=[0.1, 0.3, 0.5, 0.7, 0.9],
        dr=0.1,
        device=device
    )

    instatran_wo_M_S = SpatialTemporalTransformer2(
        d_model=10,
        d_embedding=3,
        cate_dims=[16, 32, 24],
        spatial_structure=None,
        num_cv=16,
        seq_len=48,
        num_targets=1,
        tau=12,
        quantile=[0.1, 0.3, 0.5, 0.7, 0.9],
        dr=0.1,
        device=device
    )


    instatran_w_tft_decoder = SpatialTemporalTransformer(
        d_model=30,
        d_embedding=5,
        cate_dims=[16, 32, 24],
        spatial_structure=sps,
        num_cv=16,
        seq_len=48,
        num_targets=1,
        tau=12,
        quantile=[0.1, 0.3, 0.5, 0.7, 0.9],
        dr=0.1,
        device=device
    )
    
    
    for i, model in enumerate([instatran_wo_sps, instatran_parallel, instatran_wo_M_S, instatran_w_tft_decoder]):
        
        qr = QuantileRisk(args.tau, quantile_levels, num_targets, device)

        
        model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        for _ in range(args.epochs):        

            _ = train(model, train_loader, qr, optimizer, device)

        if i == 0: 
            torch.save(model.state_dict(), '../assets/InstaTran_wo_sps.pth') 
            
        elif i == 1: 
            torch.save(model.state_dict(), '../assets/InstaTran_parallel.pth') 
            
        elif i == 2:
            torch.save(model.state_dict(), '../assets/InstaTran_wo_M_S.pth') 
            
        else:
            torch.save(model.state_dict(), '../assets/InstaTran_w_tft_decoder.pth')
            

if __name__ == '__main__':
    main()
