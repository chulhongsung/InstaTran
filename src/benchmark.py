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
from models import DeepAR, MQRnn, TemporalFusionTransformer

import argparse

parser = argparse.ArgumentParser(description='hyperparams')
parser.add_argument('--epochs', required=False, default=200, type=int)
parser.add_argument('--bs', required=False, default=500, type=int)
parser.add_argument('--lr', required=False, default=0.001, type=float)
parser.add_argument('--seed', required=False, default=7, type=int)
parser.add_argument('--d_emb', required=False, default=3, type=int)
parser.add_argument('--seq_len', required=False, default=48, type=int)
parser.add_argument('--tau', required=False, default=12, type=int)
parser.add_argument('--d_model', required=False, default=30, type=int)
parser.add_argument('--n_layer', required=False, default=1, type=int)
parser.add_argument('--dr', required=False, default=0.1, type=float)
parser.add_argument('--model', required=False, default="DeepAR", choices=("DeepAR", "MQRnn", "TFT"), type=str)

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    
def evaluate(model, loader, criterion, device):
    model.eval()
    
    total_loss = []
    
    for batch in loader:
        conti_input, cate_input, future_input, true_y = batch 
        
        conti_input = conti_input.to(device)
        cate_input = cate_input.to(device)
        future_input = future_input.to(device)
        true_y = true_y.to(device)
        
        pred = model(conti_input, cate_input, future_input)
        
        loss = criterion(true_y, pred)
        
        total_loss.append(loss)
        
        return sum(total_loss)/len(total_loss) 
    
def main():
    
    df_train_total = pd.read_csv("./data/df_train_total.csv")
    df_merged = pd.read_csv("./data/df_merged.csv")
    
    train_conti_input, train_cate_input, train_future_input, train_label = generate_ts_data(df_train_total, df_merged, input_seq_len=args.seq_len, tau=args.tau)

    torch.manual_seed(args.seed)
        
    
    train_dataset = TensorDataset(torch.FloatTensor(train_conti_input), torch.LongTensor(train_cate_input), torch.LongTensor(train_future_input), torch.FloatTensor(train_label))
    
    
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.bs)
    

    if args.model == "DeepAR":
        model = DeepAR(
                d_input=16, 
                d_embedding=args.d_emb, 
                n_embedding=[16, 32, 24], 
                d_model=args.d_model, 
                num_targets=1, 
                n_layers=args.n_layer,
                dr=args.dr
            )
        
        criterion = NegativeGaussianLogLikelihood(device)
    
    elif args.model == "TFT":
        
        quanilte_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        model = TemporalFusionTransformer(
            d_model=args.d_model,
            d_embedding=args.d_emb,
            cate_dims=[16, 32, 24],
            num_cv=16,
            seq_len=args.seq_len,
            num_targets=1,
            tau=args.tau,
            quantile=quanilte_levels,
            dr=args.dr,
            device=device
        )
                
        criterion = QuantileRisk(args.tau, quanilte_levels, 4, device)
    
    elif args.model == "MQRnn":
    
        model = MQRnn(
                d_input=16,
                d_embedding=args.d_emb,
                n_embedding=[16, 32, 24],
                d_model=args.d_model,
                tau=args.tau,
                num_targets=1,
                num_quantiles=5,
                n_layers=args.n_layer,
                dr=args.dr
            )
        
        quanilte_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        criterion = QuantileRisk(args.tau, quanilte_levels, 4, device)
    
    model.to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
       
    for _ in range(args.epochs):        

        train(model, train_loader, criterion, optimizer, device)
    
    torch.save(model.state_dict(), '../assets/{}.pth'.format(args.model))     
    
if __name__ == '__main__':
    main()