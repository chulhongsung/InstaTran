import torch
import torch.nn as nn
import numpy as np 
import pandas as pd

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

def generate_eval_ts(df, label_df, input_seq_len=48, tau=12):
    
    col_labels =  ['wl_1018680'] 
    
    tmp_df = np.array(df.loc[df['year'] == 2021, :])
    tmp_label_df = np.array(label_df.loc[label_df['year'] == 2021, col_labels])
    
    n = tmp_df.shape[0] - input_seq_len - tau 
    
    tmp_conti_input = tmp_df[:, 4:] 
    tmp_cate_input = tmp_df[:, 1:4] 
    
    conti_input = np.zeros((n, input_seq_len, tmp_conti_input.shape[1]), dtype=np.float32)
    cate_input = np.zeros((n, input_seq_len, tmp_cate_input.shape[1]), dtype=np.float32)
    future_input = np.zeros((n, tau, tmp_cate_input.shape[1]), dtype=np.float32)
    label = np.zeros((n, tau, len(col_labels)))

    past_input = np.zeros((n, input_seq_len, len(col_labels)), dtype=np.float32)
    label = np.zeros((n, tau, len(col_labels)))

    for j in range(n):
        past_input[j, :, :] = tmp_label_df[j:(j+input_seq_len), :]/1000
        conti_input[j, :, :] = tmp_conti_input[j:(j+input_seq_len), :]
        cate_input[j, :, :] = tmp_cate_input[j:(j+input_seq_len), :]
        future_input[j, :, :] = tmp_cate_input[(j+input_seq_len):(j+input_seq_len+tau), :]
        label[j, :, :] = tmp_label_df[(j+input_seq_len):(j+input_seq_len+tau), :]/1000

    return conti_input, cate_input, future_input, label, past_input

def scaled_dot_product_attention(q, k, v, d_model, mask, device):
    matmul_qk = torch.matmul(q, k.transpose(-1, -2))
    dk = torch.tensor(d_model).to(device)

    scaled_attention_logits = matmul_qk / torch.sqrt(dk)
    
    if mask is not None:
        scaled_attention_logits = scaled_attention_logits + ((1 - mask) * -1e9)

    attention_weights = nn.Softmax(dim=-1)(scaled_attention_logits)
    
    output = torch.matmul(attention_weights, v)
    
    return output, attention_weights

class QuantileRisk(nn.Module):
    def __init__(self, tau, quantile, num_targets, device):
        super(QuantileRisk, self).__init__()
        self.quantile = quantile
        self.device = device
        self.q_arr = torch.tensor(self.quantile).float().unsqueeze(0).unsqueeze(-1).repeat(1, 1, tau).unsqueeze(1).repeat(1, num_targets, 1, 1).to(self.device)
    
    def forward(self, true, pred):
        true_rep = true.unsqueeze(-1).repeat(1, 1, 1, len(self.quantile)).permute(0, 2, 3, 1).to(self.device)
        pred = pred.permute(0, 2, 3, 1)

        ql = torch.maximum(self.q_arr * (true_rep - pred), (1-self.q_arr)*(pred - true_rep))
        
        return ql.mean()


class NegativeGaussianLogLikelihood(nn.Module):
    def __init__(self, device):
        super(NegativeGaussianLogLikelihood, self).__init__()
        import math
        self.pi = torch.tensor(math.pi).float().to(device)
        
    def forward(self, true, pred):
        mu, sigma = pred
        return (torch.square(true - mu)/(2*sigma) + torch.log(2*self.pi*sigma)/2).mean()

def gaussian_quantile(mu, sigma):
    from scipy.stats import norm
    batch_size, _, _ = mu.shape

    mu = mu.detach().cpu().numpy()
    sigma = sigma.detach().cpu().numpy()

    total_output = []
    
    for q in [0.1, 0.3, 0.5, 0.7, 0.9]:
        tmp_output = []

        for i in range(batch_size):
            tmp_output.append(norm.ppf(q, loc=mu[i], scale=sigma[i])[np.newaxis, ...])
            
        total_output.append(np.concatenate(tmp_output, axis=0)[..., np.newaxis])
    
    return np.concatenate(total_output, axis=-1)