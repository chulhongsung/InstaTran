import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F

from utils import *

class ContiFeatureEmbedding(nn.Module):
    def __init__(self, d_embedding, num_rv):
        super(ContiFeatureEmbedding, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(1, d_embedding) for _ in range(num_rv)])
        
    def forward(self, x):
        tmp_feature_list = []
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            tmp_feature = l(x[:, :, i:i+1])
            tmp_feature_list.append(tmp_feature)
            
        return torch.stack(tmp_feature_list, axis=-1).transpose(-1, -2)

class CateFeatureEmbedding(nn.Module):
    def __init__(self, d_embedding, n_embedding):
        super(CateFeatureEmbedding, self).__init__()
        self.embedding_layers = nn.ModuleList([nn.Embedding(n, d_embedding) for n in n_embedding]) 
        
    def forward(self, x):
        
        tmp_feature_list = []
        
        for i, l in enumerate(self.embedding_layers):
            tmp_feature = l(x[:, :, i:i+1])
            tmp_feature_list.append(tmp_feature)
            
        return torch.cat(tmp_feature_list, axis=-2)
    
class GLULN(nn.Module):
    def __init__(self, d_model):
        super(GLULN, self).__init__()
        self.linear1 = nn.LazyLinear(d_model)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.LazyLinear(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, y):
        return self.layer_norm(torch.mul(self.sigmoid(self.linear1(x)), self.linear2(x)) + y)

class GatedResidualNetwork(nn.Module):
    def __init__(self, d_model, dr):
        super(GatedResidualNetwork, self).__init__()
        self.linear1 = nn.LazyLinear(d_model)
        self.dropout1 = nn.Dropout(dr)
        self.linear2 = nn.LazyLinear(d_model)
        self.dropout2 = nn.Dropout(dr)
        self.elu = nn.ELU()
        self.gluln = GLULN(d_model)
        
    def forward(self, x):
        eta_2 = self.dropout1(self.linear1(x))
        eta_1 = self.elu(self.dropout2(self.linear2(eta_2)))
        grn_output = self.gluln(eta_1, eta_2)
        
        return grn_output

class VariableSelectionNetwork(nn.Module):
    def __init__(self, d_model, d_input, dr):
        super(VariableSelectionNetwork, self).__init__()
        self.v_grn = GatedResidualNetwork(d_input, dr)
        self.softmax = nn.Softmax(dim=-1)
        self.xi_grn = nn.ModuleList([GatedResidualNetwork(d_model, dr) for _ in range(d_input)])
        
    def forward(self, xi):
        Xi = xi.reshape(xi.size(0), xi.size(1), -1)
        weights = self.softmax(self.v_grn(Xi)).unsqueeze(-1)
        
        tmp_xi_list = []
        for i, l in enumerate(self.xi_grn):
            tmp_xi = l(xi[:, :, i:i+1])
            tmp_xi_list.append(tmp_xi)
        xi_list = torch.cat(tmp_xi_list, axis=-2)
        
        combined = torch.matmul(weights.transpose(3, 2), xi_list).squeeze()
        
        return combined, weights
    
class SpatialStructureAttention(nn.Module):
    def __init__(self, d_model, spatial_mask, dr, device):
        super(SpatialStructureAttention, self).__init__()
        self.d_model = d_model
        self.device = device
        self.dr = dr
        self.spatial_mask = spatial_mask.to(self.device) if spatial_mask is not None else spatial_mask
                
        self.wq = nn.LazyLinear(self.d_model)
        self.dropout1 = nn.Dropout(self.dr)
        
        self.wk = nn.LazyLinear(self.d_model)
        self.dropout2 = nn.Dropout(self.dr)

        self.wv = nn.LazyLinear(self.d_model)
        self.dropout3 = nn.Dropout(self.dr)

        self.linear = nn.Linear(self.d_model, self.d_model)
            
    def forward(self, q, k, v):
        q_ = self.wq(q)
        q_ = self.dropout1(q_)
        
        k_ = self.wk(k)
        k_ = self.dropout2(k_)
        
        v_ = self.wv(v)    
        v_ = self.dropout3(v_)
        
        attention_output, attention_weight = scaled_dot_product_attention(q_, k_, v_, self.d_model, self.spatial_mask, self.device)
        
        output = self.linear(attention_output)
        
        return output, attention_weight
    
class TemporalStructureAttention(nn.Module):
    def __init__(self, d_model, seq_len, device):
        super(TemporalStructureAttention, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.device = device
        self.wq = nn.LazyLinear(self.d_model)
        self.wk = nn.LazyLinear(self.d_model)
        self.wv = nn.LazyLinear(self.d_model)
        
        self.linear = nn.Linear(self.d_model, self.d_model)
        self.temporal_mask = torch.tril(torch.ones([seq_len, seq_len]), diagonal=0).to(self.device)
        
    def forward(self, q, k, v):
        batch_size = q.size(0)
        num_var = v.size(-2)
        
        q_ = self.wq(q)
        k_ = self.wk(k)
        v_ = v.view(batch_size, self.seq_len, -1)
        
        attention_output_, attention_weight = scaled_dot_product_attention(q_, k_, v_, self.d_model, self.temporal_mask, self.device)
        
        attention_output = attention_output_.view(batch_size, self.seq_len, num_var, -1)
        output = self.linear(attention_output)
        
        return output, attention_weight
    
class SpatialTemporalDecoder(nn.Module):
    def __init__(self, d_model, dr, seq_len, device):
        super(SpatialTemporalDecoder, self).__init__()
        self.d_model = d_model
        self.dr = dr
        self.seq_len = seq_len
        self.device = device
        
        self.lstm_future = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model,
            batch_first=True)
        
        self.gluln1 = GLULN(self.d_model)
        self.mha = nn.MultiheadAttention(self.d_model, 1, batch_first=True, dropout=self.dr)
        self.gluln2 = GLULN(self.d_model)
        self.temporal_mask = torch.triu(torch.ones([self.seq_len, self.seq_len], dtype=torch.bool), diagonal=1).to(self.device)
        
    def forward(self, vsn_observed, vsn_future):
        time_step = vsn_observed.size(1) + vsn_future.size(1)
        
        future_lstm_output, (_, _) = self.lstm_future(
            vsn_future, (vsn_observed[:, -1:, ...].transpose(1, 0).contiguous(), vsn_observed[:, -1:, ...].transpose(1,0).contiguous())
            )
                
        lstm_hidden = torch.cat([vsn_observed, future_lstm_output], dim=1)
        input_vsn = torch.cat([vsn_observed, vsn_future], dim=1)
        
        glu_phi_list = []
        
        for t in range(time_step):
            tmp_phi_t = self.gluln1(lstm_hidden[:, t:t+1, :], input_vsn[:, t:t+1, :])
            glu_phi_list.append(tmp_phi_t)
        
        glu_phi = torch.cat(glu_phi_list, axis=1)
        B, decoder_attention = self.mha(query=glu_phi, key=glu_phi, value=glu_phi, attn_mask=self.temporal_mask)
        glu_delta_list = [] 

        for j in range(time_step):
            tmp_delta_t = self.gluln2(B[:, j:j+1, :], glu_phi[:, j:j+1, :])
            glu_delta_list.append(tmp_delta_t)

        glu_delta = torch.cat(glu_delta_list, dim=1)
        
        return glu_delta, glu_phi, decoder_attention

class SpatialTemporalDecoder2(nn.Module):
    def __init__(self, d_model, dr, seq_len, device):
        super(SpatialTemporalDecoder2, self).__init__()
        self.d_model = d_model
        self.dr = dr
        self.seq_len = seq_len
        self.device = device
                
        self.gluln1 = GLULN(self.d_model)
        self.mha = nn.MultiheadAttention(self.d_model, 1, batch_first=True, dropout=self.dr)
        self.gluln2 = GLULN(self.d_model)
        self.temporal_mask = torch.triu(torch.ones([self.seq_len, self.seq_len], dtype=torch.bool), diagonal=1).to(self.device)
        
    def forward(self, vsn_local_future, vsn_global):
        time_step = vsn_local_future.size(1)
                                    
        glu_phi_list = []
        
        for t in range(time_step):
            tmp_phi_t = self.gluln1(vsn_local_future[:, t:t+1, :], vsn_global[:, t:t+1, :])
            glu_phi_list.append(tmp_phi_t)
        
        glu_phi = torch.cat(glu_phi_list, axis=1)
        B, decoder_attention = self.mha(query=glu_phi, key=glu_phi, value=glu_phi, attn_mask=self.temporal_mask)
        glu_delta_list = [] 

        for j in range(time_step):
            tmp_delta_t = self.gluln2(B[:, j:j+1, :], glu_phi[:, j:j+1, :])
            glu_delta_list.append(tmp_delta_t)

        glu_delta = torch.cat(glu_delta_list, dim=1)
        
        return glu_delta, glu_phi, decoder_attention
    
class TemporalFusionDecoder(nn.Module):
    def __init__(self, d_model, dr, seq_len, device):
        super(TemporalFusionDecoder, self).__init__()
        self.d_model = d_model
        self.dr = dr
        self.seq_len = seq_len
        self.device = device

        self.lstm_obs = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model,
            batch_first=True)

        
        self.lstm_future = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model,
            batch_first=True)
        
        self.gluln1 = GLULN(self.d_model)
        self.mha = nn.MultiheadAttention(self.d_model, 1, batch_first=True, dropout=self.dr)
        self.gluln2 = GLULN(self.d_model)
        self.temporal_mask = torch.triu(torch.ones([self.seq_len, self.seq_len], dtype=torch.bool), diagonal=1).to(self.device)
        
    def forward(self, vsn_observed, vsn_future):
        time_step = vsn_observed.size(1) + vsn_future.size(1)

        future_lstm_output, (_, _) = self.lstm_obs(vsn_observed)
        
        future_lstm_output, (_, _) = self.lstm_future(
            vsn_future, (vsn_observed[:, -1:, ...].transpose(1, 0).contiguous(), vsn_observed[:, -1:, ...].transpose(1,0).contiguous())
            )
                
        lstm_hidden = torch.cat([vsn_observed, future_lstm_output], dim=1)
        input_vsn = torch.cat([vsn_observed, vsn_future], dim=1)
        
        glu_phi_list = []
        
        for t in range(time_step):
            tmp_phi_t = self.gluln1(lstm_hidden[:, t:t+1, :], input_vsn[:, t:t+1, :])
            glu_phi_list.append(tmp_phi_t)
        
        glu_phi = torch.cat(glu_phi_list, axis=1)
        B, decoder_attention = self.mha(query=glu_phi, key=glu_phi, value=glu_phi, attn_mask=self.temporal_mask)
        glu_delta_list = [] 

        for j in range(time_step):
            tmp_delta_t = self.gluln2(B[:, j:j+1, :], glu_phi[:, j:j+1, :])
            glu_delta_list.append(tmp_delta_t)

        glu_delta = torch.cat(glu_delta_list, dim=1)
        
        return glu_delta, glu_phi, decoder_attention
    
class PointWiseFeedForward(nn.Module):
    def __init__(self, d_model, dr):
        super(PointWiseFeedForward, self).__init__()
        self.grn = GatedResidualNetwork(d_model, dr)
        self.gluln = GLULN(d_model)
        
    def forward(self, delta, phi):
        time_step = delta.size(1)
        
        grn_varphi_list = []
        
        for t in range(time_step):
            tmp_grn_varphi = self.grn(delta[:, t:t+1, :])
            grn_varphi_list.append(tmp_grn_varphi)
            
        grn_varphi = torch.cat(grn_varphi_list, dim=1)
        
        varphi_tilde_list = []
        
        for t in range(time_step):
            tmp_varphi_tilde = self.gluln(grn_varphi[:, t:t+1, :], phi[:, t:t+1, :])
            varphi_tilde_list.append(tmp_varphi_tilde)
            
        varphi = torch.cat(varphi_tilde_list, dim=1)
        
        return varphi        

class TargetFeatureLayer(nn.Module):
    def __init__(self, d_model, num_target):
        super(TargetFeatureLayer, self).__init__()
        self.target_feature_linears = nn.ModuleList([nn.LazyLinear(d_model) for _ in range(num_target)])
    
    def forward(self, varphi):
        target_feature_list = []
        
        for _, l in enumerate(self.target_feature_linears):
            tmp_feature = l(varphi)
            target_feature_list.append(tmp_feature.unsqueeze(-2))
            
        return torch.cat(target_feature_list, dim=-2)

class QuantileOutput(nn.Module):
    def __init__(self, tau, quantile):
        super(QuantileOutput, self).__init__()
        self.tau = tau
        self.quantile_linears = nn.ModuleList([nn.LazyLinear(1) for _ in range(len(quantile))])
        
    def forward(self, varphi):
        total_output_list = []
        
        for _, l in enumerate(self.quantile_linears):
            tmp_quantile_list = []
            
            for t in range(self.tau-1):
                tmp_quantile = l(varphi[:, (-self.tau + t) : (-self.tau + t + 1), ...])
                tmp_quantile_list.append(tmp_quantile)
            
            tmp_quantile_list.append(l(varphi[:, -1:, ...]))
            
            total_output_list.append(torch.cat(tmp_quantile_list, dim=1))
            
        return torch.cat(total_output_list, dim=-1)
    
### Benchmark models

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

class DeepARDecoder(nn.Module):
    def __init__(self, d_input, d_embedding, n_embedding, d_model, num_targets, n_layers=3, dr=0.1):
        super(DeepARDecoder, self).__init__()
        self.n_layers = n_layers
        self.embedding_layers = nn.ModuleList([nn.Embedding(n, d_embedding) for n in n_embedding]) 
        self.lstm = nn.LSTM(d_input + len(n_embedding) * d_embedding, d_model, n_layers, dropout=dr, batch_first=True)
        
        self.linear1 = nn.Linear(d_model, num_targets)
        self.linear2 = nn.Linear(d_model, num_targets)
        self.dropout = nn.Dropout(dr)
        self.softplus = nn.Softplus(beta=2)
        
    def forward(self, future, hidden, cell):
        tmp_feature_list = []
        
        for i, l in enumerate(self.embedding_layers):
            tmp_feature = l(future[:, :, i:i+1])
            tmp_feature_list.append(tmp_feature)
        
        tau = future.size(1)    
        
        emb_output = torch.cat(tmp_feature_list, axis=-2)
        emb_output = emb_output.view(future.size(0), tau, -1) # (batch_size, tau, len(n_embedding) * d_embedding)
        
        lstm_output = []
   
        for t in range(tau):
            lstm_input = torch.cat([hidden[self.n_layers-1:self.n_layers].transpose(1, 0), emb_output[:, t:t+1, :]], axis=-1)
            output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
            lstm_output.append(output)
        
        lstm_output = torch.cat(lstm_output, axis=1)
        
        mu = self.linear1(lstm_output)
        sigma = self.softplus(self.linear2(lstm_output))
        
        return mu, sigma
    
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
        
        return output.transpose(2, 1) # (batch_size, tau, num_targets, num_quantiles)

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


def nconv(x, A):
    return torch.einsum('ncvl,vw->ncwl', (x, A)).contiguous()

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
        
        # Dilated Causal Conv -> WeightedNorm -> ReLU -> Dropout -> ... (논문의 Residual block 구조와 동일)
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
        # broadcast add
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