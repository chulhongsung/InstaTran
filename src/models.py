import torch.nn as nn

from layers import *

class SpatialTemporalTransformer(nn.Module):
    """
    Spatio-Temporal Transformer

    Args:
        d_model (int): hidden feature size
        d_embedding (int): embedding size of continuous and categorical variables
        cate_dims (list): the number of category of categorical variables, e.g., [12, 31, 24] (month, day, hour)
        spatial_structure (list): masking index for get_spatial_mask
        d_input (int): the number of input variables (conti + cate)
        num_cv (int): the number of continuous variables 
        seq_len (int): the length of input sequence
        num_targets (int): the number of targets
        tau (int): the length of target sequence
        quantile (list): target quantile levels
        num_heads (int): the number of heads in multihead attenion layer
        dr (float): dropout rate
    """
    def __init__(
        self, 
        d_model,
        d_embedding,
        cate_dims,
        spatial_structure,
        num_cv,
        seq_len,
        num_targets,
        tau,
        quantile,
        dr,
        device
    ):
        super(SpatialTemporalTransformer, self).__init__()
        self.confe = ContiFeatureEmbedding(d_embedding, num_cv)
        self.catfe = CateFeatureEmbedding(d_embedding, cate_dims)
        
        self.vsn1 = VariableSelectionNetwork(d_model, num_cv, dr)
        self.vsn2 = VariableSelectionNetwork(d_model, num_cv, dr)
        self.vsn3 = VariableSelectionNetwork(d_model, len(cate_dims), dr)
        
        self.ssa1 = SpatialStructureAttention(d_model, spatial_structure, dr, device)
        self.ssa2 = SpatialStructureAttention(d_model, spatial_structure, dr, device)
        
        self.tsa = TemporalStructureAttention(d_model, seq_len=seq_len, device=device)
        self.std = SpatialTemporalDecoder(d_model, dr, seq_len=seq_len + tau, device=device)
        self.pwff = PointWiseFeedForward(d_model, dr)
        self.tfl = TargetFeatureLayer(d_model, num_targets)
        self.qo = QuantileOutput(tau, quantile)
        
    def forward(self, conti_input, cate_input, future_input):
        
        confe_output = self.confe(conti_input) # (batch_size, seq_len, num_cv, d_embedding)
        catfe_output = self.catfe(cate_input)  # (batch_size, seq_len, num_cate, d_embedding)
        
        cate_feature = catfe_output.mean(dim=-2, keepdim=True) # (batch_size, seq_len, 1, d_embedding)

        ### Spatio-Temporal Feature Learning
        obs_feature = confe_output + cate_feature 
        
        ssa_output1, ssa_weight1 = self.ssa1(obs_feature, obs_feature, obs_feature) # (batch_size, seq_len, num_cv, d_model), (batch_size, seq_len, num_cv, num_cv)
        
        vsn_output1, _ = self.vsn1(ssa_output1) # (batch_size, seq_len, d_model)
        
        tsa_output, tsa_weight = self.tsa(vsn_output1, vsn_output1, ssa_output1) # (batch_size, seq_len, num_cv, d_model)
        
        ssa_output2, ssa_weight2 = self.ssa2(tsa_output, tsa_output, tsa_output) # (batch_size, seq_len, num_cv, d_model)
        
        vsn_output2, feature_importance1 = self.vsn2(ssa_output2) # (batch_size, seq_len, d_model)
        
        ### Decoder step
        
        future_embedding = self.catfe(future_input) # (batch_size, tau, num_cate, d_embedding)
        
        vsn_future_output, feature_importance2 = self.vsn3(future_embedding) # (batch_size, tau, d_model)
        
        std_delta, std_phi, dec_weights = self.std(vsn_output2, vsn_future_output) # (batch_size, seq_len+tau, d_model), (batch_size, seq_len+tau, d_model)
        
        varphi = self.pwff(std_delta, std_phi) # (batch_size, seq_len+tau, d_model)
        
        ### Target Quantile Forecasting
        
        tfl_output = self.tfl(varphi) # (batch_size, seq_len+tau, num_target, d_model)
        
        output = self.qo(tfl_output) # (batch_size, tau, num_target, quantile)
        
        return output, ssa_weight1, ssa_weight2, tsa_weight, dec_weights, feature_importance1, feature_importance2 # (batch_size, tau, num_target, quantile), (batch_size, seq_len, num_cv, num_cv), (batch_size, seq_len, num_cv, num_cv), (batch_size, seq_len, seq_len), (batch_size, seq_len+tau, seq_len+tau)

class SpatialTemporalParallelTransformer(nn.Module):
    """
    Spatio-Temporal Parallel Transformer

    Args:
        d_model (int): hidden feature size
        d_embedding (int): embedding size of continuous and categorical variables
        cate_dims (list): the number of category of categorical variables, e.g., [12, 31, 24] (month, day, hour)
        spatial_structure (list): masking index for get_spatial_mask
        d_input (int): the number of input variables (conti + cate)
        num_cv (int): the number of continuous variables 
        seq_len (int): the length of input sequence
        num_targets (int): the number of targets
        tau (int): the length of target sequence
        quantile (list): target quantile levels
        num_heads (int): the number of heads in multihead attenion layer
        dr (float): dropout rate
    """
    def __init__(
        self, 
        d_model,
        d_embedding,
        cate_dims,
        spatial_structure,
        num_cv,
        seq_len,
        num_targets,
        tau,
        quantile,
        dr,
        device
    ):
        super(SpatialTemporalParallelTransformer, self).__init__()
        self.confe = ContiFeatureEmbedding(d_embedding, num_cv)
        self.catfe = CateFeatureEmbedding(d_embedding, cate_dims)
            
        self.ssa = SpatialStructureAttention(1, spatial_structure, dr, device)
        
        self.mha1 = nn.MultiheadAttention(num_cv, 1, batch_first=True, dropout=dr)
        self.mha2 = nn.MultiheadAttention(num_cv, 1, batch_first=True, dropout=dr)
        self.vsn = VariableSelectionNetwork(num_cv, len(cate_dims), dr)
        
        # self.std = SpatialTemporalDecoder(d_model, dr, seq_len=seq_len + tau, device=device)
        self.pwff = PointWiseFeedForward(d_model, dr)
        self.tfl = TargetFeatureLayer(d_model, num_targets)
        self.qo = QuantileOutput(tau, quantile)
        
        self.temporal_mask1 = torch.triu(torch.ones([seq_len, seq_len], dtype=torch.bool), diagonal=1).to(device)
        self.temporal_mask2 = torch.triu(torch.ones([seq_len+tau, seq_len+tau], dtype=torch.bool), diagonal=1).to(device)
        
    def forward(self, conti_input, cate_input, future_input):
        
        confe_output = self.confe(conti_input) # (batch_size, seq_len, num_cv, d_embedding)
        catfe_output = self.catfe(cate_input)  # (batch_size, seq_len, num_cate, d_embedding)
        
        cate_feature = catfe_output.mean(dim=-2, keepdim=True) # (batch_size, seq_len, 1, d_embedding)

        ### Spatio-Temporal Feature Learning
        obs_feature = confe_output + cate_feature 
        
        ssa_output1, ssa_weight = self.ssa(obs_feature, obs_feature, obs_feature) # (batch_size, seq_len, num_cv, 1), (batch_size, seq_len, num_cv, num_cv)
                
        tsa_output, tsa_weight = self.mha1(conti_input, conti_input, conti_input, attn_mask=self.temporal_mask1) # (batch_size, seq_len, num_cv), (batch_size, seq_len, seq_len) 
        
        encoder_output = tsa_output + ssa_output1.squeeze() # (batch_size, seq_len, num_cv)
        
        ### Decoder step
        
        future_embedding = self.catfe(future_input) # (batch_size, tau, num_cate, d_embedding)
        
        vsn_future_output, _ = self.vsn(future_embedding) # (batch_size, tau, num_cv)
        
        decoder_input = torch.cat([encoder_output, vsn_future_output], axis=1) # (batch_size, seq_len+tau, num_cv)
        
        dec_output, dec_weights = self.mha2(decoder_input, decoder_input, decoder_input, attn_mask=self.temporal_mask2) # (batch_size, seq_len+tau, d_model), (batch_size, seq_len+tau, d_model)
        
        
        ### Target Quantile Forecasting
        
        tfl_output = self.tfl(dec_output) # (batch_size, seq_len+tau, num_target, d_model)
        
        output = self.qo(tfl_output) # (batch_size, tau, num_target, quantile)
        
        return output, ssa_weight, ssa_weight, tsa_weight, dec_weights, _, _ # (batch_size, tau, num_target, quantile), (batch_size, seq_len, num_cv, num_cv), (batch_size, seq_len, num_cv, num_cv), (batch_size, seq_len, seq_len), (batch_size, seq_len+tau, seq_len+tau)


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer

    Args:
        d_model (int): hidden feature size
        d_embedding (int): embedding size of continuous and categorical variables
        cate_dims (list): the number of category of categorical variables, e.g., [12, 31, 24] (month, day, hour)
        spatial_structure (list): masking index for get_spatial_mask
        d_input (int): the number of input variables (conti + cate)
        num_cv (int): the number of continuous variables 
        seq_len (int): the length of input sequence
        num_targets (int): the number of targets
        tau (int): the length of target sequence
        quantile (list): target quantile levels
        num_heads (int): the number of heads in multihead attenion layer
        dr (float): dropout rate
    """
    def __init__(
        self, 
        d_model,
        d_embedding,
        cate_dims,
        num_cv,
        seq_len,
        num_targets,
        tau,
        quantile,
        dr,
        device
    ):
        super(TemporalFusionTransformer, self).__init__()
        self.confe = ContiFeatureEmbedding(d_embedding, num_cv)
        self.catfe = CateFeatureEmbedding(d_embedding, cate_dims)
        self.num_var = num_cv + len(cate_dims)
        self.vsn1 = VariableSelectionNetwork(d_model, self.num_var, dr)
        self.vsn2 = VariableSelectionNetwork(d_model, len(cate_dims), dr)

        self.tfd = TemporalFusionDecoder(d_model, dr, seq_len=seq_len + tau, device=device)
        self.pwff = PointWiseFeedForward(d_model, dr)
        self.tfl = TargetFeatureLayer(d_model, num_targets)
        self.qo = QuantileOutput(tau, quantile)
        
    def forward(self, conti_input, cate_input, future_input):
        
        confe_output = self.confe(conti_input) # (batch_size, seq_len, num_cv, d_embedding)
        catfe_output = self.catfe(cate_input)  # (batch_size, seq_len, num_cate, d_embedding)

        ### Encoder
        obs_feature = torch.cat([confe_output, catfe_output], axis=-2)  # (batch_size, seq_len, num_cv + num_cate, d_embedding)
        x1, _  = self.vsn1(obs_feature) # (batch_size, seq_len, d_model)
        
        ### Decoder
        future_embedding = self.catfe(future_input) # (batch_size, tau, num_cate, d_embedding)
        x2, _ = self.vsn2(future_embedding) # (batch_size, tau, d_embedding)
        delta, glu_phi, decoder_weights = self.tfd(x1, x2) # # (batch_size, seq_len+tau, d_model)
        varphi = self.pwff(delta, glu_phi) # (batch_size, seq_len+tau, num_target, d_model)
        tfl_output = self.tfl(varphi)  # (batch_size, seq_len+tau, d_model)
        output = self.qo(tfl_output) # (batch_size, tau, num_target, quantile)
        
        return output

class DeepAR(nn.Module):
    def __init__(self, d_input, d_embedding, n_embedding, d_model, num_targets, n_layers=3, dr=0.1):
        super(DeepAR, self).__init__()

        self.encoder = Encoder(
                               d_input=d_input,
                               d_embedding=d_embedding,
                               n_embedding=n_embedding,
                               d_model=d_model,
                               n_layers=n_layers,
                               dr=dr
                               )
        self.decoder = DeepARDecoder(
                                     d_input=d_model,
                                     d_embedding=d_embedding,
                                     n_embedding=n_embedding,
                                     d_model=d_model,
                                     num_targets=num_targets,
                                     n_layers=n_layers,
                                     dr=dr
                                     )

    def forward(self, conti, cate, future):
        
        encoder_hidden, encoder_cell = self.encoder(conti, cate)
        mu, sigma = self.decoder(future, encoder_hidden, encoder_cell)
        
        return mu, sigma
    
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

class InstaTran(nn.Module):
    """
    Spatio-Temporal Transformer

    Args:
        d_model (int): hidden feature size
        d_embedding (int): embedding size of continuous and categorical variables
        cate_dims (list): the number of category of categorical variables, e.g., [12, 31, 24] (month, day, hour)
        spatial_structure (list): masking index for get_spatial_mask
        d_input (int): the number of input variables (conti + cate)
        num_cv (int): the number of continuous variables 
        seq_len (int): the length of input sequence
        num_targets (int): the number of targets
        tau (int): the length of target sequence
        quantile (list): target quantile levels
        num_heads (int): the number of heads in multihead attenion layer
        dr (float): dropout rate
    """
    def __init__(
        self, 
        d_model,
        d_embedding,
        cate_dims,
        spatial_structure,
        num_cv,
        seq_len,
        num_targets,
        tau,
        quantile,
        dr,
        device
    ):
        super(InstaTran, self).__init__()
        self.confe = ContiFeatureEmbedding(d_embedding, num_cv)
        self.catfe = CateFeatureEmbedding(d_embedding, cate_dims)
        
        self.vsn1 = VariableSelectionNetwork(d_model, num_cv, dr)
        self.vsn2 = VariableSelectionNetwork(d_model, num_cv, dr)
        self.vsn3 = VariableSelectionNetwork(d_model, len(cate_dims), dr)
        self.vsn4 = VariableSelectionNetwork(d_model, seq_len, dr)
        
        self.ssa1 = SpatialStructureAttention(d_model, spatial_structure, dr, device)
        self.ssa2 = SpatialStructureAttention(d_model, spatial_structure, dr, device)
        
        self.tsa = TemporalStructureAttention(d_model, seq_len=seq_len, device=device)
        self.mha = nn.MultiheadAttention(d_model, 1, batch_first=True, dropout=dr)
        self.temporal_mask = torch.triu(torch.ones([seq_len+tau, seq_len+tau], dtype=torch.bool), diagonal=1).to(device)
        # self.std = SpatialTemporalDecoder(d_model, dr, seq_len=seq_len + tau, device=device)
        self.pwff = PointWiseFeedForward(d_model, dr)
        self.tfl = TargetFeatureLayer(d_model, num_targets)
        self.qo = QuantileOutput(tau, quantile)
        
    def forward(self, conti_input, cate_input, future_input):
        
        confe_output = self.confe(conti_input) # (batch_size, seq_len, num_cv, d_embedding)
        catfe_output = self.catfe(cate_input)  # (batch_size, seq_len, num_cate, d_embedding)
        
        cate_feature = catfe_output.mean(dim=-2, keepdim=True) # (batch_size, seq_len, 1, d_embedding)

        ### Spatio-Temporal Feature Learning
        obs_feature = confe_output + cate_feature 
        
        ssa_output1, ssa_weight1 = self.ssa1(obs_feature, obs_feature, obs_feature) # (batch_size, seq_len, num_cv, d_model), (batch_size, seq_len, num_cv, num_cv)
        
        vsn_output1, _ = self.vsn1(ssa_output1) # (batch_size, seq_len, d_model)
        
        tsa_output, tsa_weight = self.tsa(vsn_output1, vsn_output1, ssa_output1) # (batch_size, seq_len, num_cv, d_model)
        
        ssa_output2, ssa_weight2 = self.ssa2(tsa_output, tsa_output, tsa_output) # (batch_size, seq_len, num_cv, d_model)
        
        vsn_output2, feature_importance1 = self.vsn2(ssa_output2) # (batch_size, seq_len, d_model)
        
        ### Decoder step
        
        future_embedding = self.catfe(future_input) # (batch_size, tau, num_cate, d_embedding)
        
        future_feature, future_importance  = self.vsn3(future_embedding) # (batch_size, tau, d_model)
            
        global_context, global_importance = self.vsn4(vsn_output2.unsqueeze(1)) # (batch_size, d_model)

        local_context = global_context.unsqueeze(1) + future_feature # (batch_size, tau, d_model)
        
        total_context = torch.cat([vsn_output2, local_context], axis=1) # (batch_size, seq_len+tau, d_model)
        
        B, decoder_attention = self.mha(query=total_context, key=total_context, value=total_context, attn_mask=self.temporal_mask) # (batch_size, seq_len+tau, d_model)
                
        ### Target Quantile Forecasting
        
        tfl_output = self.tfl(B) # (batch_size, seq_len+tau, num_target, d_model)
        
        output = self.qo(tfl_output) # (batch_size, tau, num_target, quantile)
        
        return output, ssa_weight1, ssa_weight2, tsa_weight, decoder_attention, feature_importance1, future_importance # (batch_size, tau, num_target, quantile), (batch_size, seq_len, num_cv, num_cv), (batch_size, seq_len, num_cv, num_cv), (batch_size, seq_len, seq_len), (batch_size, seq_len+tau, seq_len+tau)


class STALSTM(nn.Module):
    """ 
        Ding et al., 2020 (Neurocomputing)
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
        
        return torch.cat(total_output_list, dim=-1).unsqueeze(-2), alpha, beta
    
class HSDSTM(nn.Module):
    """Deng et al, 2023 (Stochastic Environmental Research and Risk Assessment)
    """
    def __init__(self, 
                 adj,
                 input_size,
                 seq_len,
                 num_channels,
                 node_dim,
                 dropout,
                 num_levels,
                 tau,
                 num_quantiles
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
        
        return torch.cat(total_output_list, dim=-1).unsqueeze(-2)
    
    
class SpatialTemporalTransformer2(nn.Module):
    """
    Spatio-Temporal Transformer

    Args:
        d_model (int): hidden feature size
        d_embedding (int): embedding size of continuous and categorical variables
        cate_dims (list): the number of category of categorical variables, e.g., [12, 31, 24] (month, day, hour)
        spatial_structure (list): masking index for get_spatial_mask
        d_input (int): the number of input variables (conti + cate)
        num_cv (int): the number of continuous variables 
        seq_len (int): the length of input sequence
        num_targets (int): the number of targets
        tau (int): the length of target sequence
        quantile (list): target quantile levels
        num_heads (int): the number of heads in multihead attenion layer
        dr (float): dropout rate
    """
    def __init__(
        self, 
        d_model,
        d_embedding,
        cate_dims,
        spatial_structure,
        num_cv,
        seq_len,
        num_targets,
        tau,
        quantile,
        dr,
        device
    ):
        super(SpatialTemporalTransformer2, self).__init__()
        self.confe = ContiFeatureEmbedding(d_embedding, num_cv)
        self.catfe = CateFeatureEmbedding(d_embedding, cate_dims)
        
        self.vsn1 = VariableSelectionNetwork(d_model, num_cv, dr)
        self.vsn2 = VariableSelectionNetwork(d_model, num_cv, dr)
        self.vsn3 = VariableSelectionNetwork(d_model, len(cate_dims), dr)
        self.vsn4 = VariableSelectionNetwork(d_model, seq_len, dr)
        
        self.ssa1 = SpatialStructureAttention(d_model, spatial_structure, dr, device)
        self.ssa2 = SpatialStructureAttention(d_model, spatial_structure, dr, device)
        
        self.tsa = TemporalStructureAttention(d_model, seq_len=seq_len, device=device)
        self.mha = nn.MultiheadAttention(d_model, 1, batch_first=True, dropout=dr)
        self.temporal_mask = torch.triu(torch.ones([seq_len+tau, seq_len+tau], dtype=torch.bool), diagonal=1).to(device)
        # self.std = SpatialTemporalDecoder(d_model, dr, seq_len=seq_len + tau, device=device)
        self.pwff = PointWiseFeedForward(d_model, dr)
        self.tfl = TargetFeatureLayer(d_model, num_targets)
        self.qo = QuantileOutput(tau, quantile)
        
    def forward(self, conti_input, cate_input, future_input):
        
        confe_output = self.confe(conti_input) # (batch_size, seq_len, num_cv, d_embedding)
        catfe_output = self.catfe(cate_input)  # (batch_size, seq_len, num_cate, d_embedding)
        
        cate_feature = catfe_output.mean(dim=-2, keepdim=True) # (batch_size, seq_len, 1, d_embedding)

        ### Spatio-Temporal Feature Learning
        obs_feature = confe_output + cate_feature 
        
        ssa_output1, ssa_weight1 = self.ssa1(obs_feature, obs_feature, obs_feature) # (batch_size, seq_len, num_cv, d_model), (batch_size, seq_len, num_cv, num_cv)
        
        vsn_output1, _ = self.vsn1(ssa_output1) # (batch_size, seq_len, d_model)
        
        tsa_output, tsa_weight = self.tsa(vsn_output1, vsn_output1, ssa_output1) # (batch_size, seq_len, num_cv, d_model)
        
        ssa_output2, ssa_weight2 = self.ssa2(tsa_output, tsa_output, tsa_output) # (batch_size, seq_len, num_cv, d_model)
        
        vsn_output2, feature_importance1 = self.vsn2(ssa_output2) # (batch_size, seq_len, d_model)
        
        ### Decoder step
        
        future_embedding = self.catfe(future_input) # (batch_size, tau, num_cate, d_embedding)
        
        future_feature, future_importance  = self.vsn3(future_embedding) # (batch_size, tau, d_model)
            
        global_context, global_importance = self.vsn4(vsn_output2.unsqueeze(1)) # (batch_size, d_model)

        local_context = global_context.unsqueeze(1) + future_feature # (batch_size, tau, d_model)
        
        total_context = torch.cat([vsn_output2, local_context], axis=1) # (batch_size, seq_len+tau, d_model)
        
        B, decoder_attention = self.mha(query=total_context, key=total_context, value=total_context, attn_mask=self.temporal_mask) # (batch_size, seq_len+tau, d_model)
                
        ### Target Quantile Forecasting
        
        tfl_output = self.tfl(B) # (batch_size, seq_len+tau, num_target, d_model)
        
        output = self.qo(tfl_output) # (batch_size, tau, num_target, quantile)
        
        return output, ssa_weight1, ssa_weight2, tsa_weight, decoder_attention, feature_importance1, future_importance # (batch_size, tau, num_target, quantile), (batch_size, seq_len, num_cv, num_cv), (batch_size, seq_len, num_cv, num_cv), (batch_size, seq_len, seq_len), (batch_size, seq_len+tau, seq_len+tau)


class SpatialTemporalParallelTransformer(nn.Module):
    """
    Spatio-Temporal Parallel Transformer

    Args:
        d_model (int): hidden feature size
        d_embedding (int): embedding size of continuous and categorical variables
        cate_dims (list): the number of category of categorical variables, e.g., [12, 31, 24] (month, day, hour)
        spatial_structure (list): masking index for get_spatial_mask
        d_input (int): the number of input variables (conti + cate)
        num_cv (int): the number of continuous variables 
        seq_len (int): the length of input sequence
        num_targets (int): the number of targets
        tau (int): the length of target sequence
        quantile (list): target quantile levels
        num_heads (int): the number of heads in multihead attenion layer
        dr (float): dropout rate
    """
    def __init__(
        self, 
        d_model,
        d_embedding,
        cate_dims,
        spatial_structure,
        num_cv,
        seq_len,
        num_targets,
        tau,
        quantile,
        dr,
        device
    ):
        super(SpatialTemporalParallelTransformer, self).__init__()
        self.confe = ContiFeatureEmbedding(d_embedding, num_cv)
        self.catfe = CateFeatureEmbedding(d_embedding, cate_dims)
            
        self.ssa = SpatialStructureAttention(1, spatial_structure, dr, device)
        
        self.mha1 = nn.MultiheadAttention(num_cv, 1, batch_first=True, dropout=dr)
        self.mha2 = nn.MultiheadAttention(num_cv, 1, batch_first=True, dropout=dr)
        self.vsn = VariableSelectionNetwork(num_cv, len(cate_dims), dr)
        
        # self.std = SpatialTemporalDecoder(d_model, dr, seq_len=seq_len + tau, device=device)
        self.pwff = PointWiseFeedForward(d_model, dr)
        self.tfl = TargetFeatureLayer(d_model, num_targets)
        self.qo = QuantileOutput(tau, quantile)
        
        self.temporal_mask1 = torch.triu(torch.ones([seq_len, seq_len], dtype=torch.bool), diagonal=1).to(device)
        self.temporal_mask2 = torch.triu(torch.ones([seq_len+tau, seq_len+tau], dtype=torch.bool), diagonal=1).to(device)
        
    def forward(self, conti_input, cate_input, future_input):
        
        confe_output = self.confe(conti_input) # (batch_size, seq_len, num_cv, d_embedding)
        catfe_output = self.catfe(cate_input)  # (batch_size, seq_len, num_cate, d_embedding)
        
        cate_feature = catfe_output.mean(dim=-2, keepdim=True) # (batch_size, seq_len, 1, d_embedding)

        ### Spatio-Temporal Feature Learning
        obs_feature = confe_output + cate_feature 
        
        ssa_output1, ssa_weight = self.ssa(obs_feature, obs_feature, obs_feature) # (batch_size, seq_len, num_cv, 1), (batch_size, seq_len, num_cv, num_cv)
                
        tsa_output, tsa_weight = self.mha1(conti_input, conti_input, conti_input, attn_mask=self.temporal_mask1) # (batch_size, seq_len, num_cv), (batch_size, seq_len, seq_len) 
        
        encoder_output = tsa_output + ssa_output1.squeeze() # (batch_size, seq_len, num_cv)
        
        ### Decoder step
        
        future_embedding = self.catfe(future_input) # (batch_size, tau, num_cate, d_embedding)
        
        vsn_future_output, _ = self.vsn(future_embedding) # (batch_size, tau, num_cv)
        
        decoder_input = torch.cat([encoder_output, vsn_future_output], axis=1) # (batch_size, seq_len+tau, num_cv)
        
        dec_output, dec_weights = self.mha2(decoder_input, decoder_input, decoder_input, attn_mask=self.temporal_mask2) # (batch_size, seq_len+tau, d_model), (batch_size, seq_len+tau, d_model)
        
        
        ### Target Quantile Forecasting
        
        tfl_output = self.tfl(dec_output) # (batch_size, seq_len+tau, num_target, d_model)
        
        output = self.qo(tfl_output) # (batch_size, tau, num_target, quantile)
        
        return output, ssa_weight, ssa_weight, tsa_weight, dec_weights, _, _ # (batch_size, tau, num_target, quantile), (batch_size, seq_len, num_cv, num_cv), (batch_size, seq_len, num_cv, num_cv), (batch_size, seq_len, seq_len), (batch_size, seq_len+tau, seq_len+tau)
    
    
class SpatialTemporalTransformer(nn.Module):
    """
    Spatio-Temporal Transformer

    Args:
        d_model (int): hidden feature size
        d_embedding (int): embedding size of continuous and categorical variables
        cate_dims (list): the number of category of categorical variables, e.g., [12, 31, 24] (month, day, hour)
        spatial_structure (list): masking index for get_spatial_mask
        d_input (int): the number of input variables (conti + cate)
        num_cv (int): the number of continuous variables 
        seq_len (int): the length of input sequence
        num_targets (int): the number of targets
        tau (int): the length of target sequence
        quantile (list): target quantile levels
        num_heads (int): the number of heads in multihead attenion layer
        dr (float): dropout rate
    """
    def __init__(
        self, 
        d_model,
        d_embedding,
        cate_dims,
        spatial_structure,
        num_cv,
        seq_len,
        num_targets,
        tau,
        quantile,
        dr,
        device
    ):
        super(SpatialTemporalTransformer, self).__init__()
        self.confe = ContiFeatureEmbedding(d_embedding, num_cv)
        self.catfe = CateFeatureEmbedding(d_embedding, cate_dims)
        
        self.vsn1 = VariableSelectionNetwork(d_model, num_cv, dr)
        self.vsn2 = VariableSelectionNetwork(d_model, num_cv, dr)
        self.vsn3 = VariableSelectionNetwork(d_model, len(cate_dims), dr)
        
        self.ssa1 = SpatialStructureAttention(d_model, spatial_structure, dr, device)
        self.ssa2 = SpatialStructureAttention(d_model, spatial_structure, dr, device)
        
        self.tsa = TemporalStructureAttention(d_model, seq_len=seq_len, device=device)
        self.std = SpatialTemporalDecoder(d_model, dr, seq_len=seq_len + tau, device=device)
        self.pwff = PointWiseFeedForward(d_model, dr)
        self.tfl = TargetFeatureLayer(d_model, num_targets)
        self.qo = QuantileOutput(tau, quantile)
        
    def forward(self, conti_input, cate_input, future_input):
        
        confe_output = self.confe(conti_input) # (batch_size, seq_len, num_cv, d_embedding)
        catfe_output = self.catfe(cate_input)  # (batch_size, seq_len, num_cate, d_embedding)
        
        cate_feature = catfe_output.mean(dim=-2, keepdim=True) # (batch_size, seq_len, 1, d_embedding)

        ### Spatio-Temporal Feature Learning
        obs_feature = confe_output + cate_feature 
        
        ssa_output1, ssa_weight1 = self.ssa1(obs_feature, obs_feature, obs_feature) # (batch_size, seq_len, num_cv, d_model), (batch_size, seq_len, num_cv, num_cv)
        
        vsn_output1, _ = self.vsn1(ssa_output1) # (batch_size, seq_len, d_model)
        
        tsa_output, tsa_weight = self.tsa(vsn_output1, vsn_output1, ssa_output1) # (batch_size, seq_len, num_cv, d_model)
        
        ssa_output2, ssa_weight2 = self.ssa2(tsa_output, tsa_output, tsa_output) # (batch_size, seq_len, num_cv, d_model)
        
        vsn_output2, feature_importance1 = self.vsn2(ssa_output2) # (batch_size, seq_len, d_model)
        
        ### Decoder step
        
        future_embedding = self.catfe(future_input) # (batch_size, tau, num_cate, d_embedding)
        
        vsn_future_output, feature_importance2 = self.vsn3(future_embedding) # (batch_size, tau, d_model)
        
        std_delta, std_phi, dec_weights = self.std(vsn_output2, vsn_future_output) # (batch_size, seq_len+tau, d_model), (batch_size, seq_len+tau, d_model)
        
        varphi = self.pwff(std_delta, std_phi) # (batch_size, seq_len+tau, d_model)
        
        ### Target Quantile Forecasting
        
        tfl_output = self.tfl(varphi) # (batch_size, seq_len+tau, num_target, d_model)
        
        output = self.qo(tfl_output) # (batch_size, tau, num_target, quantile)
        
        return output, ssa_weight1, ssa_weight2, tsa_weight, dec_weights, feature_importance1, feature_importance2 # (batch_size, tau, num_target, quantile), (batch_size, seq_len, num_cv, num_cv), (batch_size, seq_len, num_cv, num_cv), (batch_size, seq_len, seq_len), (batch_size, seq_len+tau, seq_len+tau)