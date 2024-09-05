#%%
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import sys
import os
sys.path
# sys.path.append(os.getcwd() + "/torch")

from utils import *
from layers import *
from models import *

import pandas as pd
import numpy as np

import seaborn as sns
from ing_theme_matplotlib import mpl_style # pip install ing_theme_matplotlib 
import matplotlib as mpl
import matplotlib.pyplot as plt

df_train_total = pd.read_csv("../data/df_train_total.csv")
df_test_total = pd.read_csv("../data/df_test_total.csv")
df_merged = pd.read_csv("../data/df_merged.csv")
df_train_total
train_conti_input, train_cate_input, train_future_input, train_label = generate_ts_data(df_train_total, df_merged)
test_conti_input, test_cate_input, test_future_input, test_label = generate_ts_data(df_test_total, df_merged)

eval_a, eval_b, eval_c, eval_d, eval_e = generate_eval_ts(df_test_total, df_merged, input_seq_len=48, tau=12)
#%%
eval_conti = torch.FloatTensor(eval_a)
eval_cate = torch.LongTensor(eval_b)
eval_future = torch.LongTensor(eval_c)
eval_label = torch.FloatTensor(eval_d)
eval_past_label = torch.FloatTensor(eval_e)
#%%
eval_conti.shape
eval_cate.shape
eval_future.shape
#%%
tft = TemporalFusionTransformer(
    d_model=30,
    d_embedding=5,
    cate_dims=[16, 32, 24],
    num_cv=16,
    seq_len=48,
    num_targets=1,
    tau=12,
    quantile=[0.1, 0.3, 0.5, 0.7, 0.9],
    dr=0.1,
    device=device
)
                
deepar = DeepAR(
        d_input=16, 
        d_embedding=3, 
        n_embedding=[16, 32, 24], 
        d_model=30, 
        num_targets=1, 
        n_layers=3,
        dr=0.1
    )

mqrnn = MQRnn(
        d_input=16,
        d_embedding=1,
        n_embedding=[16, 32, 24],
        d_model=5,
        tau=12,
        num_targets=1,
        num_quantiles=5,
        n_layers=3,
        dr=0.1
    )

# deepar.load_state_dict(torch.load("/Users/chulhongsung/Desktop/lab/working_paper/ts/water_level_forecasting/assets/save_weights/DeepAR_23-0226-1412_final.pth", map_location='cpu'))
deepar.load_state_dict(torch.load("/Users/shong/Desktop/ts/water_level_forecasting/assets/save_weights/DeepAR_23-0328-1651_best.pth", map_location='cpu'))
tft.load_state_dict(torch.load("/Users/shong/Desktop/ts/water_level_forecasting/assets/save_weights/TFT_23-0124-1331.pth", map_location="cpu"))
mqrnn.load_state_dict(torch.load("/Users/shong/Desktop/ts/water_level_forecasting/assets/save_weights/MQRnn_23-0123-2038.pth", map_location="cpu"))
# %%
deepar.eval()
output_deepar = deepar(eval_conti, eval_cate, eval_future)
output_deepar_mu, output_deepar_sigma = output_deepar
output_deepar_mu.detach().cpu().numpy()
output_deepar_mu.shape
deepar_output = gaussian_quantile(output_deepar_mu, output_deepar_sigma)
deepar_output.shape
#%%
tft.eval()
tft_output, decoder_weights = tft(eval_conti, eval_cate, eval_future)

#%%
mqrnn.eval()
mqrnn_output = mqrnn(eval_conti, eval_cate, eval_future)
mqrnn_output.shape

#%%
# plot_results(eval_label[150:170], stt_recalibration_output[150:170].unsqueeze(-2), dark=False)
# plot_results(eval_label[150:170], stt_output[150:170], dark=False)
# plot_results(eval_label[150:170], tft_output[150:170], dark=False)
# plot_results(eval_label[150:170], torch.tensor(deepar_output[150:170]), dark=False)
# plot_results(eval_label[::12], mqrnn_output[::12], dark=False)
plot_results(eval_label[150:170], mqrnn_output[150:170], dark=False)
eval_label.shape
mqrnn_output.shape
#%%
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
#%%
stt = SpatialTemporalTransformer(
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
#%%
stt_no_sps = SpatialTemporalTransformer(
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
#%%
stt_parallel = SpatialTemporalParallelTransformer(
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

#%%
stt2 = SpatialTemporalTransformer2(
    d_model=10,
    d_embedding=3,
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
#%%
stt2_without = SpatialTemporalTransformer2(
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
# %%
stt2.load_state_dict(torch.load("/Users/shong/Desktop/ts/water_level_forecasting/assets/save_weights/stt_23-0224-1857_best.pth", map_location='cpu'))
# stt2.load_state_dict(torch.load("/Users/shong/Desktop/ts/water_level_forecasting/assets/save_weights/stt_23-0225-1507_best.pth", map_location='cpu'))
stt2_without.load_state_dict(torch.load("/Users/shong/Desktop/ts/water_level_forecasting/assets/save_weights/stt_without_M_S_23-0227-1356_best.pth", map_location='cpu'))
stt.load_state_dict(torch.load("/Users/shong/Desktop/ts/water_level_forecasting/assets/save_weights/stt_23-0118-2338_best.pth", map_location='cpu'))
stt_no_sps.load_state_dict(torch.load("/Users/shong/Desktop/ts/water_level_forecasting/assets/save_weights/stt_no_sps-23-0119-0205_final.pth", map_location='cpu'))
stt_parallel.load_state_dict(torch.load("/Users/shong/Desktop/ts/water_level_forecasting/assets/save_weights/stt_parallel_23-0118-1944_final.pth", map_location='cpu'))
# stt2.load_state_dict(torch.load("/Users/shong/Desktop/ts/water_level_forecasting/assets/save_weights/stt2_23-0221-1115_best.pth", map_location='cpu'))
#%%
mpl.rcParams["figure.dpi"] = 300
mpl_style(dark=False)
# %%
eval_conti = torch.FloatTensor(eval_a)
eval_cate = torch.LongTensor(eval_b)
eval_future = torch.LongTensor(eval_c)
eval_label = torch.FloatTensor(eval_d)
eval_past_label = torch.FloatTensor(eval_e)
eval_label.shape
# output, ssa_weight1, ssa_weight2, tsa_weight, dec_weights, fi1, fi2 = stt(eval_conti, eval_cate, eval_future)
stt2.eval()
output, ssa_weight1, ssa_weight2, tsa_weight, dec_weights, fi1, fi2 = stt2(eval_conti, eval_cate, eval_future)
# output_non, ssa_weight1_non, ssa_weight2_non, tsa_weight_non, dec_weights_non, fi1_non, fi2_non = non_stt(eval_conti, eval_cate, eval_future)
# output_parallel, ssa_weight1_parallel, ssa_weight2_parallel, tsa_weight_parallel, dec_weights_parallel, _, _ = parallel_stt(eval_conti, eval_cate, eval_future)
#%%
# fi1.detach().numpy()[0, ...].shape
# fi1.detach().numpy()[0, ...].squeeze().mean(axis=0).shape
batch_num = 15
quantile = 0.9
ax = plt.matshow(np.quantile(fi1.detach().numpy()[batch_num, ...].squeeze(), quantile, axis=0)[np.newaxis, ...])
plt.colorbar(ax)
#%%
ax = plt.matshow(np.quantile(fi2.detach().numpy()[batch_num, ...].squeeze(), quantile, axis=0)[np.newaxis, ...])
plt.colorbar(ax)

# ax = plt.matshow(np.quantile(fi1_non.detach().numpy()[batch_num, ...].squeeze(), quantile, axis=0)[np.newaxis, ...])
# plt.colorbar(ax)


# ax = plt.matshow(np.quantile(fi2_non.detach().numpy()[batch_num, ...].squeeze(), quantile, axis=0)[np.newaxis, ...])
# plt.colorbar(ax)

feature_idx = 13

fi1.detach().numpy()[batch_num, :, feature_idx]
eval_conti[batch_num, :, feature_idx].detach().numpy()

#%%
batch_num = 10
time_step = 15
plt.matshow(ssa_weight1[batch_num, time_step, ...].detach().numpy(), cmap='Reds')
plt.colorbar()
#%%
plt.matshow(ssa_weight2[batch_num, time_step, ...].detach().numpy(), cmap='Reds')
plt.colorbar()

plt.matshow(tsa_weight[batch_num, ...].detach().numpy(), cmap='cividis')
plt.xlabel("Time point")
plt.ylabel("Time point")
plt.colorbar()

plt.matshow((dec_weights[batch_num, ...] / dec_weights[batch_num, ...].sum(-1).unsqueeze(-1)).detach().numpy(), cmap='cividis')
plt.colorbar()

# batch_num = 15
# time_step = 15
# plt.matshow(ssa_weight1_non[batch_num, time_step, ...].detach().numpy(), cmap='Reds')
# plt.colorbar()
# plt.matshow(ssa_weight2_non[batch_num, time_step, ...].detach().numpy(), cmap='Reds')
# plt.colorbar()
# plt.matshow(tsa_weight_non[batch_num, ...].detach().numpy(), cmap='cividis')
# plt.colorbar()
# plt.matshow((dec_weights_non[batch_num, ...] / dec_weights[batch_num, ...].sum(-1).unsqueeze(-1)).detach().numpy(), cmap='cividis')
# plt.colorbar()

# batch_num = 300
# time_step = 20
# plt.matshow(ssa_weight1_parallel[batch_num, time_step, ...].detach().numpy(), cmap='Reds')
# plt.colorbar()
# plt.matshow(ssa_weight2_parallel[batch_num, time_step, ...].detach().numpy(), cmap='Reds')
# plt.colorbar()
# plt.matshow(tsa_weight_parallel[batch_num, ...].detach().numpy(), cmap='cividis')
# plt.colorbar()
# plt.matshow((dec_weights_parallel[batch_num, ...] / dec_weights[batch_num, ...].sum(-1).unsqueeze(-1)).detach().numpy(), cmap='cividis')
# plt.colorbar()


# plt.plot(fi1_non.detach().numpy()[batch_num, :, feature_idx].squeeze())
# plt.plot(eval_conti[batch_num, :, feature_idx].detach().numpy())
#%% Rainy season
eval_conti = torch.FloatTensor(eval_a[1500:1650])
eval_cate = torch.LongTensor(eval_b[1500:1650])
eval_future = torch.LongTensor(eval_c[1500:1650])
eval_label = torch.FloatTensor(eval_d[1500:1650])
eval_past_label = torch.FloatTensor(eval_e[1500:1650])
#%%
# output, ssa_weight1, ssa_weight2, tsa_weight, dec_weights, fi1, fi2 = stt(eval_conti, eval_cate, eval_future)
stt2.eval()
output, ssa_weight1, ssa_weight2, tsa_weight, dec_weights, fi1, fi2 = stt2(eval_conti, eval_cate, eval_future)
# output
# eval_label
#%%
stt_no_sps.eval()
output_no_sps, ssa_weight1_no_sps, ssa_weight2_no_sps, tsa_weight_no_sps, dec_weights_no_sps, fi1_no_sps, fi2_no_sps = stt_no_sps(eval_conti, eval_cate, eval_future)
#%%
plt.plot(eval_conti[::48, :, 0].reshape(-1))

batch_num = 120
time_step = 3
# new_feature_importance = ssa_weight2.mean(dim=-2).unsqueeze(-1) * fi1
# new_feature_importance.shape

## Prep1 시각화
eval_conti[::48, :, 0].reshape(-1)
plt.plot(eval_conti[::48, :, 0].reshape(-1))


plt.matshow(ssa_weight1[batch_num, time_step, ...].detach().numpy(), cmap='Reds')
plt.colorbar()
#%%
plt.matshow(ssa_weight2[50, 0, ...].detach().numpy(), cmap='Reds')
plt.colorbar()
plt.xlabel("Variable index")
plt.ylabel("Variable index")
#%%
plt.matshow(ssa_weight2_no_sps[50, 0, ...].detach().numpy(), cmap='Reds')
plt.colorbar()
plt.xlabel("Variable index")
plt.ylabel("Variable index")
#%%
plt.matshow(ssa_weight2[10, 15, ...].detach().numpy(), cmap='Reds')
plt.colorbar()
plt.xlabel("Variable index")
plt.ylabel("Variable index")
#%%
plt.matshow(ssa_weight2_no_sps[10, 15, ...].detach().numpy(), cmap='Reds')
plt.colorbar()
plt.xlabel("Variable index")
plt.ylabel("Variable index")
#%%
plt.matshow(ssa_weight2[5, 15, ...].detach().numpy(), cmap='Reds')
plt.colorbar()
plt.xlabel("Variable index")
plt.ylabel("Variable index")
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 8].squeeze().reshape(-1), label=r"Observation $P_2$").set(xlabel="Time points")
plt.matshow(tsa_weight[50, ...].detach().numpy(), cmap='cividis')
plt.colorbar()
# plt.matshow((dec_weights[batch_num, ...] / dec_weights[batch_num, ...].sum(-1).unsqueeze(-1)).detach().numpy(), cmap='cividis')
# plt.colorbar()
#%%
ax = sns.lineplot(eval_conti.cpu()[30, :, 2].squeeze().reshape(-1), label=r"Observation $P_2$").set(xlabel="Time points")
sns.lineplot(fi1[10, :, 15, ...].squeeze().reshape(-1).detach().numpy(), label=r"Observation $P_2$").set(xlabel="Time points")
# plt.matshow(tsa_weight[50, ...].detach().numpy(), cmap='cividis')
# plt.colorbar()

plt.matshow(ssa_weight2[50, 4, ...].detach().numpy(), cmap='Reds')
plt.colorbar()
plt.xlabel("Variable index")
plt.ylabel("Variable index")

plt.matshow(tsa_weight[10, ...].detach().numpy(), cmap='cividis')
plt.colorbar()
plt.xlabel("Time point")
plt.ylabel("Time point")

plt.matshow(tsa_weight[30, ...].detach().numpy(), cmap='cividis')
plt.colorbar()
plt.xlabel("Time point")
plt.ylabel("Time point")

plt.matshow(dec_weights[100, ...].detach().numpy(), cmap='cividis')
plt.colorbar()
plt.xlabel("Time point")
plt.ylabel("Time point")

dec_weights

#%% Time points 50
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(tsa_weight[50, ...].detach().numpy(), cmap='cividis')
fig.colorbar(cax)
ax.set_xticks([0, 10, 20, 23, 30, 40])
ax.set_yticks([0, 10, 20, 30, 40])
ax.set_xticklabels(['0', '10', '20', '23', '30', '40'])
ax.set_yticklabels(['0', '10', '20', '30', '40'])
ax.set_xlabel("Time points")
ax.set_ylabel("Time points")
#%% TSA weights consistency visualization
fig, (ax1, ax2) = plt.subplots(1, 2)

cax1 = ax1.matshow(tsa_weight[30, ...].detach().numpy(), cmap='cividis')
ax1.set_xticks([0, 10, 23, 30, 40, 47])
ax1.set_yticks([0, 10, 20, 30, 40, 47])
ax1.set_xticklabels(['0', '10', '23', '30', '40', '47'])
ax1.set_yticklabels(['0', '10', '20', '30', '40', '47'])
ax1.set_xlabel("Time points")
ax1.set_ylabel("Time points")
# ax1.axvline(x=22, ls=':', lw=1.0)

cax2 = ax2.matshow(tsa_weight[50, ...].detach().numpy(), cmap='cividis')
ax2.set_xticks([3, 10, 20, 30, 40, 47])
ax2.set_yticks([0, 10, 20, 30, 40, 47])
ax2.set_xticklabels(['23', '30', '40', '50', '60', '67'])
ax2.set_yticklabels(['20', '30', '40', '50', '60', '67'])
ax2.set_xlabel("Time points")
# ax2.axvline(x=2, ls=':', lw=1.0)

# ax2.set_ylabel("Time points")
cbar_ax = fig.add_axes([0.95, 0.15, 0.03, 0.7])
fig.colorbar(cax2, cax=cbar_ax)
#%%
plt.show()
plt.matshow(tsa_weight[50, ...].detach().numpy(), )
plt.colorbar()
plt.xlabel("Time point")
plt.ylabel("Time point")

plt.matshow(tsa_weight[70, ...].detach().numpy(), cmap='cividis')
plt.colorbar()
plt.xlabel("Time point")
plt.ylabel("Time point")
#%%
# ax = plt.matshow(np.quantile(new_feature_importance[batch_num, ...].detach().numpy().squeeze(), quantile, axis=0)[np.newaxis, ...])
# plt.colorbar(ax)
#%%
# TFT vsn 
eval_conti = torch.FloatTensor(eval_a[1500:1650])
eval_cate = torch.LongTensor(eval_b[1500:1650])
eval_future = torch.LongTensor(eval_c[1500:1650])
eval_label = torch.FloatTensor(eval_d[1500:1650])
eval_past_label = torch.FloatTensor(eval_e[1500:1650])


tft.eval()
confe_output = tft.confe(eval_conti) # (batch_size, seq_len, num_cv, d_embedding)
catfe_output = tft.catfe(eval_cate)  # (batch_size, seq_len, num_cate, d_embedding)
obs_feature = torch.cat([confe_output, catfe_output], axis=-2)  # (batch_size, seq_len, num_cv + num_cate, d_embedding)
x1, tft_vsn_output  = tft.vsn1(obs_feature) # (batch_size, seq_len, d_model)

# VSN quantile 
stt2.eval()
# stt_output, ssa_weight1, ssa_weight2, tsa_weight, dec_weights, fi1, fi2  = stt(eval_conti, eval_cate, eval_future)
stt_output, ssa_weight1, ssa_weight2, tsa_weight, dec_weights, fi1, fi2  = stt2(eval_conti, eval_cate, eval_future)
#%%
stt_no_sps.eval()
output_no_sps, ssa_weight1_no_sps, ssa_weight2_no_sps, tsa_weight_no_sps, dec_weights_no_sps, fi1_no_sps, fi2_no_sps = stt_no_sps(eval_conti, eval_cate, eval_future)
#%%
stt_parallel.eval()
output_parallel, ssa_weight1_parallel, ssa_weight2_parallel, tsa_weight_parallel, dec_weights_parallel, fi1_parallel, fi2_parallel = stt_parallel(eval_conti, eval_cate, eval_future)
#%%
ax = plt.matshow(fi1.detach().numpy().squeeze().reshape(-1, 16)[100:101,])
plt.colorbar(ax)
plt.yticks([])
# fi1

quantile = 0.9
ax = plt.matshow(np.quantile(fi1.detach().numpy().squeeze().reshape(-1, 16), quantile, axis=0)[np.newaxis, ...])
plt.colorbar(ax)
plt.yticks([])

# ax = plt.matshow(np.quantile(tft_vsn_output.detach().numpy().squeeze()[..., :16].reshape(-1, 16), quantile, axis=0)[np.newaxis, ...])
# plt.colorbar(ax)
# plt.yticks([])

# np.quantile(tft_vsn_output.detach().numpy().squeeze()[..., :16].reshape(-1, 16), 0.1, axis=0)
# np.quantile(tft_vsn_output.detach().numpy().squeeze()[..., :16].reshape(-1, 16), 0.5, axis=0)
# np.quantile(tft_vsn_output.detach().numpy().squeeze()[..., :16].reshape(-1, 16), 0.9, axis=0)

# tft_vsn_output.detach().numpy().squeeze()[..., :16].reshape(-1, 16).mean(axis=0)
# tft_vsn_output.detach().numpy().squeeze()[..., :16].reshape(-1, 16).std(axis=0)
#%%
fi1.detach().numpy().squeeze().reshape(-1, 16).mean(axis=0)
fi1.detach().numpy().squeeze().reshape(-1, 16).std(axis=0)
np.quantile(fi1.detach().numpy().squeeze().reshape(-1, 16), 0.1, axis=0)
np.quantile(fi1.detach().numpy().squeeze().reshape(-1, 16), 0.5, axis=0)
np.quantile(fi1.detach().numpy().squeeze().reshape(-1, 16), 0.9, axis=0)

#%%
#
quantile = 0.5
ax = plt.matshow(np.quantile(fi1.detach().numpy().squeeze().reshape(-1, 16), quantile, axis=0)[np.newaxis, ...])
plt.colorbar(ax)
plt.yticks([])

ax = plt.matshow(np.quantile(tft_vsn_output.detach().numpy().squeeze()[..., :16].reshape(-1, 16), quantile, axis=0)[np.newaxis, ...])
plt.colorbar(ax)
plt.yticks([])

#
quantile = 0.1
ax = plt.matshow(np.quantile(fi1.detach().numpy().squeeze().reshape(-1, 16), quantile, axis=0)[np.newaxis, ...])
plt.colorbar(ax)
plt.yticks([])

ax = plt.matshow(np.quantile(tft_vsn_output.detach().numpy().squeeze()[..., :16].reshape(-1, 16), quantile, axis=0)[np.newaxis, ...])
plt.colorbar(ax)
plt.yticks([])
#%%
sns.histplot(tft_vsn_output.detach().numpy().squeeze()[..., :16].reshape(-1, 16)[:, 0],  alpha=0.7, fill=True, stat='probability', binwidth=0.02, label='Prep1')
sns.histplot(tft_vsn_output.detach().numpy().squeeze()[..., :16].reshape(-1, 16)[:, 1],  alpha=0.7, fill=True, stat='probability', binwidth=0.02, label='Prep2')
sns.histplot(tft_vsn_output.detach().numpy().squeeze()[..., :16].reshape(-1, 16)[:, 2],  alpha=0.7, fill=True, stat='probability', binwidth=0.02, label='Prep3')
sns.histplot(tft_vsn_output.detach().numpy().squeeze()[..., :16].reshape(-1, 16)[:, 3],  alpha=0.7, fill=True, stat='probability', binwidth=0.02, label='TL')
sns.histplot(tft_vsn_output.detach().numpy().squeeze()[..., :16].reshape(-1, 16)[:, 4],  alpha=0.7, fill=True, stat='probability', binwidth=0.02, label='SWL')
sns.histplot(tft_vsn_output.detach().numpy().squeeze()[..., :16].reshape(-1, 16)[:, 5],  alpha=0.7, fill=True, stat='probability', binwidth=0.02, label='IF')
sns.histplot(tft_vsn_output.detach().numpy().squeeze()[..., :16].reshape(-1, 16)[:, 6],  alpha=0.7, fill=True, stat='probability', binwidth=0.02, label='SWF')
sns.histplot(tft_vsn_output.detach().numpy().squeeze()[..., :16].reshape(-1, 16)[:, 7],  alpha=0.7, fill=True, stat='probability', binwidth=0.02, label='JUS')
sns.histplot(tft_vsn_output.detach().numpy().squeeze()[..., :16].reshape(-1, 16)[:, 8],  alpha=0.7, fill=True, stat='probability', binwidth=0.02, label='OF')
sns.histplot(tft_vsn_output.detach().numpy().squeeze()[..., :16].reshape(-1, 16)[:, 9],  alpha=0.7, fill=True, stat='probability', binwidth=0.02, label='WF(CD)')
sns.histplot(tft_vsn_output.detach().numpy().squeeze()[..., :16].reshape(-1, 16)[:, 10], alpha=0.7, fill=True, stat='probability', binwidth=0.02, label='FW(CD)')
sns.histplot(tft_vsn_output.detach().numpy().squeeze()[..., :16].reshape(-1, 16)[:, 11], alpha=0.7, fill=True, stat='probability', binwidth=0.02, label='WL(JS)')
sns.histplot(tft_vsn_output.detach().numpy().squeeze()[..., :16].reshape(-1, 16)[:, 12], alpha=0.7, fill=True, stat='probability', binwidth=0.02, label='WL(HG)')
sns.histplot(tft_vsn_output.detach().numpy().squeeze()[..., :16].reshape(-1, 16)[:, 13], alpha=0.7, fill=True, stat='probability', binwidth=0.02, label='FW(HG)')
sns.histplot(tft_vsn_output.detach().numpy().squeeze()[..., :16].reshape(-1, 16)[:, 14], alpha=0.7, fill=True, stat='probability', binwidth=0.02, label='WL(HJ)')
sns.histplot(tft_vsn_output.detach().numpy().squeeze()[..., :16].reshape(-1, 16)[:, 15], alpha=0.7, fill=True, stat='probability', binwidth=0.02, label='FW(HJ)')
#%%
tft_vsn_output.detach().numpy().squeeze()[..., :16].reshape(-1, 16).mean(axis=0)
tft_vsn_output.detach().numpy().squeeze()[..., :16].reshape(-1, 16).std(axis=0)
#%%
sns.histplot(fi1.detach().numpy().squeeze().reshape(-1, 16)[:, 0],  alpha=0.7, fill=True, stat='probability', binwidth=0.02)
sns.histplot(fi1.detach().numpy().squeeze().reshape(-1, 16)[:, 1],  alpha=0.7, fill=True, stat='probability', binwidth=0.02)
sns.histplot(fi1.detach().numpy().squeeze().reshape(-1, 16)[:, 2],  alpha=0.7, fill=True, stat='probability', binwidth=0.02)
sns.histplot(fi1.detach().numpy().squeeze().reshape(-1, 16)[:, 3],  alpha=0.7, fill=True, stat='probability', binwidth=0.02)
sns.histplot(fi1.detach().numpy().squeeze().reshape(-1, 16)[:, 4],  alpha=0.7, fill=True, stat='probability', binwidth=0.02)
sns.histplot(fi1.detach().numpy().squeeze().reshape(-1, 16)[:, 5],  alpha=0.7, fill=True, stat='probability', binwidth=0.02)
sns.histplot(fi1.detach().numpy().squeeze().reshape(-1, 16)[:, 6],  alpha=0.7, fill=True, stat='probability', binwidth=0.02)
sns.histplot(fi1.detach().numpy().squeeze().reshape(-1, 16)[:, 7],  alpha=0.7, fill=True, stat='probability', binwidth=0.02)
sns.histplot(fi1.detach().numpy().squeeze().reshape(-1, 16)[:, 8],  alpha=0.7, fill=True, stat='probability', binwidth=0.02)
sns.histplot(fi1.detach().numpy().squeeze().reshape(-1, 16)[:, 9],  alpha=0.7, fill=True, stat='probability', binwidth=0.02)
sns.histplot(fi1.detach().numpy().squeeze().reshape(-1, 16)[:, 10], alpha=0.7, fill=True, stat='probability', binwidth=0.02)
sns.histplot(fi1.detach().numpy().squeeze().reshape(-1, 16)[:, 11], alpha=0.7, fill=True, stat='probability', binwidth=0.02)
sns.histplot(fi1.detach().numpy().squeeze().reshape(-1, 16)[:, 12], alpha=0.7, fill=True, stat='probability', binwidth=0.02)
sns.histplot(fi1.detach().numpy().squeeze().reshape(-1, 16)[:, 13], alpha=0.7, fill=True, stat='probability', binwidth=0.02)
sns.histplot(fi1.detach().numpy().squeeze().reshape(-1, 16)[:, 14], alpha=0.7, fill=True, stat='probability', binwidth=0.02)
sns.histplot(fi1.detach().numpy().squeeze().reshape(-1, 16)[:, 15], alpha=0.7, fill=True, stat='probability', binwidth=0.02)
#%%
fi1.detach().numpy().squeeze().reshape(-1, 16).mean(axis=0)
fi1.detach().numpy().squeeze().reshape(-1, 16).std(axis=0)
#%%
# sns.lineplot(eval_conti.cpu()[::48, :, 0].squeeze().reshape(-1)[30:], label='Prep1').set(xlabel="Time index")
# sns.lineplot(eval_conti.cpu()[::48, :, 1].squeeze().reshape(-1)[30:], label='Prep2')
# sns.lineplot(eval_conti.cpu()[::48, :, 2].squeeze().reshape(-1)[30:], label='Prep3')
# sns.lineplot(eval_conti.cpu()[::48, :, 3].squeeze().reshape(-1)[30:], label='Ganghwa')
# sns.lineplot(eval_conti.cpu()[::48, :, 4].squeeze().reshape(-1)[30:], label='Paldang wl')
# sns.lineplot(eval_conti.cpu()[::48, :, 5].squeeze().reshape(-1)[30:], label='Paldang in')
# sns.lineplot(eval_conti.cpu()[::48, :, 6].squeeze().reshape(-1)[30:], label='Paldang sfw')
eval_conti = torch.FloatTensor(eval_a[1500:1650])
eval_cate = torch.LongTensor(eval_b[1500:1650])
eval_future = torch.LongTensor(eval_c[1500:1650])
eval_label = torch.FloatTensor(eval_d[1500:1650])
eval_past_label = torch.FloatTensor(eval_e[1500:1650])
#%%
#%%
stt2.eval()
stt_output, ssa_weight1, ssa_weight2, tsa_weight, dec_weights, fi1, fi2  = stt2(eval_conti, eval_cate, eval_future)
# %%
deepar.eval()
output_deepar = deepar(eval_conti, eval_cate, eval_future)
output_deepar_mu, output_deepar_sigma = output_deepar
output_deepar_mu.detach().cpu().numpy()
output_deepar_mu.shape

deepar_output = gaussian_quantile(output_deepar_mu, output_deepar_sigma)
deepar_output.shape
#%%
tft.eval()
tft_output = tft(eval_conti, eval_cate, eval_future)

#%%
mqrnn.eval()
mqrnn_output = mqrnn(eval_conti, eval_cate, eval_future)
mqrnn_output.shape
#%% 결과 example plotting
plot_past_prediction_results2(eval_label, stt_output, eval_past_label, batch_num=90)
plot_past_prediction_results2(eval_label, tft_output, eval_past_label, batch_num=90)
plot_past_prediction_results2(eval_label, mqrnn_output, eval_past_label, batch_num=90)
plot_past_prediction_results2(eval_label, deepar_output, eval_past_label, batch_num=90)
#%% 5월 결과 plotting
eval_conti = torch.FloatTensor(eval_a[:744, ...])
eval_cate = torch.LongTensor(eval_b[:744, ...])
eval_future = torch.LongTensor(eval_c[:744, ...])
eval_label = torch.FloatTensor(eval_d[:744, ...])
# eval_past_label = torch.FloatTensor(eval_e[:744, ...])
#%% 6월 결과 plotting
eval_conti = torch.FloatTensor(eval_a[744:1464, ...])
eval_cate = torch.LongTensor(eval_b[744:1464, ...])
eval_future = torch.LongTensor(eval_c[744:1464, ...])
eval_label = torch.FloatTensor(eval_d[744:1464, ...])
# eval_past_label = torch.FloatTensor(eval_e[744:1464, ...])
#%% 7월 결과
eval_conti = torch.FloatTensor(eval_a[1464:2268, ...])
eval_cate = torch.LongTensor(eval_b[1464:2268, ...])
eval_future = torch.LongTensor(eval_c[1464:2268, ...])
eval_label = torch.FloatTensor(eval_d[1464:2268, ...])
# eval_past_label = torch.FloatTensor(eval_e[1464:2268, ...])
#%% 8월
eval_conti = torch.FloatTensor(eval_a[2268:3012, ...])
eval_cate = torch.LongTensor(eval_b[2268:3012, ...])
eval_future = torch.LongTensor(eval_c[2268:3012, ...])
eval_label = torch.FloatTensor(eval_d[2268:3012, ...])
# eval_past_label = torch.FloatTensor(eval_e[2268:3012, ...])
#%% 9월
eval_conti = torch.FloatTensor(eval_a[3012:3732, ...])
eval_cate = torch.LongTensor(eval_b[3012:3732, ...])
eval_future = torch.LongTensor(eval_c[3012:3732, ...])
eval_label = torch.FloatTensor(eval_d[3012:3732, ...])
# eval_past_label = torch.FloatTensor(eval_e[3012:3732, ...])
#%% 10월
eval_conti = torch.FloatTensor(eval_a[3732:, ...])
eval_cate = torch.LongTensor(eval_b[3732:, ...])
eval_future = torch.LongTensor(eval_c[3732:, ...])
eval_label = torch.FloatTensor(eval_d[3732:, ...])
# eval_past_label = torch.FloatTensor(eval_e[3732:, ...])
#%%
stt2.eval()
stt_output, ssa_weight1, ssa_weight2, tsa_weight, dec_weights, fi1, fi2  = stt2(eval_conti, eval_cate, eval_future)
# %%
deepar.eval()
output_deepar = deepar(eval_conti, eval_cate, eval_future)
output_deepar_mu, output_deepar_sigma = output_deepar
output_deepar_mu.detach().cpu().numpy()
output_deepar_mu.shape

deepar_output = gaussian_quantile(output_deepar_mu, output_deepar_sigma)
deepar_output.shape
#%%
tft.eval()
tft_output = tft(eval_conti, eval_cate, eval_future)

#%%
mqrnn.eval()
mqrnn_output = mqrnn(eval_conti, eval_cate, eval_future)
mqrnn_output.shape

#%% 결과 plotting
plot_results(eval_label, stt_output)
plot_results(eval_label, tft_output)
plot_results(eval_label, mqrnn_output)
plot_results(eval_label, deepar_output)

#%%
eval_conti = torch.FloatTensor(eval_a[1500:1650])
eval_cate = torch.LongTensor(eval_b[1500:1650])
eval_future = torch.LongTensor(eval_c[1500:1650])
eval_label = torch.FloatTensor(eval_d[1500:1650])
eval_past_label = torch.FloatTensor(eval_e[1500:1650])

stt2.eval()
stt_output, ssa_weight1, ssa_weight2, tsa_weight, dec_weights, fi1, fi2  = stt2(eval_conti, eval_cate, eval_future)

tft.eval()
confe_output = tft.confe(eval_conti) # (batch_size, seq_len, num_cv, d_embedding)
catfe_output = tft.catfe(eval_cate)  # (batch_size, seq_len, num_cate, d_embedding)
obs_feature = torch.cat([confe_output, catfe_output], axis=-2)  # (batch_size, seq_len, num_cv + num_cate, d_embedding)
x1, tft_vsn_output  = tft.vsn1(obs_feature) # (batch_size, seq_len, d_model)
#%%
# ax = sns.lineplot(eval_conti.cpu()[::48, :, 7].squeeze().reshape(-1)[30:], color='tab:orange', label='Paldang JUS').set(xlabel="Time index")
# sns.lineplot(eval_conti.cpu()[::48, :, 8].squeeze().reshape(-1)[30:], color='tab:blue', label='Paldang OF')
# # sns.lineplot(fi1.detach().cpu()[::48, :, 8, 0].reshape(-1)[30:], color='tab:green', label="Importance of Jus (STT)")
# # sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 8, 0].reshape(-1)[30:], color='tab:blue', label="Importance of Jus (TFT)")
# sns.lineplot(tsa_weight[::48].mean(dim=-2).reshape(-1).detach().cpu()[30:], color=sns.color_palette("hls", 8)[1], label="Temporal importance")
mpl.rcParams["figure.dpi"] = 200
mpl_style(dark=False)
# SMALL_SIZE = 10
# MEDIUM_SIZE = 14
# BIGGER_SIZE = 18

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
#%% P1 importance
ax = sns.lineplot(eval_conti.cpu()[::48, :, 0].squeeze().reshape(-1)[30:], label=r"Observation of $P_1$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 0, 0].reshape(-1)[30:], linestyle='--', label=r"Importance of $P_1$ (InstaTran)")
# sns.lineplot(fi1.detach().cpu()[::48, :, 11, 0].reshape(-1)[30:], label="WL(JS)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 0, 0].reshape(-1)[30:], linestyle=':', label=r"Importance of $P_1$ (TFT)")
# sns.lineplot(tsa_weight[::48].mean(dim=-2).reshape(-1).detach().cpu()[30:], color=sns.color_palette("hls", 8)[1], label="Temporal importance")
# sns.move_legend(ax, "upper left")
#%%
feature_idx = 8
importance_mat = np.zeros((150, 197))
for i in range(150):
    importance_mat[i, i:i+48] = fi1.detach().cpu()[i, :, feature_idx, 0]


tft_importance_mat = np.zeros((150, 197))
for i in range(150):
    tft_importance_mat[i, i:i+48] = tft_vsn_output.detach().cpu()[i, :, feature_idx, 0]

ax = sns.lineplot(eval_conti.cpu()[::48, :, feature_idx].squeeze().reshape(-1)[30:], label=r"Observation of $P_1$").set(xlabel="Time points")
sns.lineplot(np.nanmean(np.where(importance_mat==0.0, np.nan, importance_mat), axis=0)[30:-5], linestyle='--', label=r"Importance of $P_1$ (InstaTran)")
sns.lineplot(np.nanmean(np.where(tft_importance_mat==0.0, np.nan, tft_importance_mat), axis=0)[30:-5], linestyle=':', label=r"Importance of $P_1$ (TFT)")
#%%
importance_mat = np.zeros((150, 197))
for i in range(150):
    importance_mat[i, i:i+48] = fi1.detach().cpu()[i, :, 0, 0]


tft_importance_mat = np.zeros((150, 197))
for i in range(150):
    tft_importance_mat[i, i:i+48] = tft_vsn_output.detach().cpu()[i, :, 0, 0]

ax = sns.lineplot(eval_conti.cpu()[::48, :, 2].squeeze().reshape(-1)[30:], label=r"Observation $P_3$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 2, 0].reshape(-1)[30:],  linestyle='--', label=r"Importance of $P_3$ (InstaTran)")
# sns.lineplot(fi1.detach().cpu()[::48, :, 11, 0].reshape(-1)[:30], label="WL(JS)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 2, 0].reshape(-1)[30:], linestyle=':', label=r"Importance of $P_3$ (TFT)")
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 3].squeeze().reshape(-1)[30:], label=r"Observation of $WL(B_5)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 3, 0].reshape(-1)[30:], label=r"Importance of $WL(B_5)$ (InstaTran)")
# sns.lineplot(fi1.detach().cpu()[::48, :, 11, 0].reshape(-1)[30:], label="WL(JS)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 3, 0].reshape(-1)[30:],label=r"Importance of $WL(B_5)$ (TFT)")
#%% JUS importance
ax = sns.lineplot(eval_conti.cpu()[::48, :, 7].squeeze().reshape(-1)[30:], label=r"Observation of $JUS$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 7, 0].reshape(-1)[30:], label=r"Importance of $JUS$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 7, 0].reshape(-1)[30:],label=r"Importance of $JUS$ (TFT)")
#%% OF importance
ax = sns.lineplot(eval_conti.cpu()[::48, :, 8].squeeze().reshape(-1)[30:], label=r"Observation of $OF$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 8, 0].reshape(-1)[30:], linestyle='--', label=r"Importance of $OF$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 8, 0].reshape(-1)[30:], linestyle=':', label=r"Importance of $OF$ (TFT)")
#%% STR importance
ax = sns.lineplot(eval_conti.cpu()[::48, :, 6].squeeze().reshape(-1)[30:], label=r"Observation of $STR$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 6, 0].reshape(-1)[30:], label="Importance of $STR$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 6, 0].reshape(-1)[30:], label="Importance of $STR$ (TFT)")
#%% Hangang WL importance
ax = sns.lineplot(eval_conti.cpu()[::48, :, 12].squeeze().reshape(-1)[30:], label=r"Observation of $WL(B_3)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 12, 0].reshape(-1)[30:], label="Importance of $WL(B_3)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 12, 0].reshape(-1)[30:], label="Importance of $WL(B_3)$ (TFT)")
#%% Hangju WL importance
ax = sns.lineplot(eval_conti.cpu()[::48, :, 14].squeeze().reshape(-1)[30:], label=r"Observation of $WL(B_4)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 14, 0].reshape(-1)[30:], label="Importance of $WL(B_4)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 14, 0].reshape(-1)[30:], label="Importance of $WL(B_4)$ (TFT)")
#%%
#%% Jamsu WL importance
ax = sns.lineplot(eval_conti.cpu()[::48, :, 11].squeeze().reshape(-1)[30:], label=r"Observation of $WL(B_2)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 11, 0].reshape(-1)[30:], linestyle='--', label="Importance of $WL(B_2)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 11, 0].reshape(-1)[30:], linestyle=':', label="Importance of $WL(B_2)$ (TFT)")
#%%
#%% Jamsu WL importance
ax = sns.lineplot(eval_conti.cpu()[::48, :, 3].squeeze().reshape(-1)[30:], label=r"Observation of $WL(B_2)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 3, 0].reshape(-1)[30:], linestyle='--', label="Importance of $WL(B_2)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 3, 0].reshape(-1)[30:], linestyle=':', label="Importance of $WL(B_2)$ (TFT)")
#%% STT vs TFT
# eval_conti = torch.FloatTensor(eval_a[::12,...])
# eval_cate = torch.LongTensor(eval_b[::12,...])
# eval_future = torch.LongTensor(eval_c[::12,...])
# eval_label = torch.FloatTensor(eval_d[::12,...])
# eval_past_label = torch.FloatTensor(eval_e[::12,...])

# eval_conti = torch.FloatTensor(eval_a[1500:1650])
# eval_cate = torch.LongTensor(eval_b[1500:1650])
# eval_future = torch.LongTensor(eval_c[1500:1650])
# eval_label = torch.FloatTensor(eval_d[1500:1650])
# eval_past_label = torch.FloatTensor(eval_e[1500:1650])

eval_conti = torch.FloatTensor(eval_a)
eval_cate = torch.LongTensor(eval_b)
eval_future = torch.LongTensor(eval_c)
eval_label = torch.FloatTensor(eval_d)
eval_past_label = torch.FloatTensor(eval_e)

# stt_output, ssa_weight1, ssa_weight2, tsa_weight, dec_weights, fi1, fi2  = stt(eval_conti, eval_cate, eval_future)

stt2.eval()
stt_output, ssa_weight1, ssa_weight2, tsa_weight, dec_weights, fi1, fi2  = stt2(eval_conti, eval_cate, eval_future)

output_no_sps, *_  = stt2_without(eval_conti, eval_cate, eval_future)

#%%
tft.eval()
confe_output = tft.confe(eval_conti) # (batch_size, seq_len, num_cv, d_embedding)
catfe_output = tft.catfe(eval_cate)  # (batch_size, seq_len, num_cate, d_embedding)
obs_feature = torch.cat([confe_output, catfe_output], axis=-2)  # (batch_size, seq_len, num_cv + num_cate, d_embedding)
x1, tft_vsn_output  = tft.vsn1(obs_feature) # (batch_size, seq_len, d_model)

ax = sns.lineplot(eval_conti.cpu()[::48, :, 7].squeeze().reshape(-1)[30:], color='tab:orange', label='Paldang Jus')
sns.lineplot(fi1.detach().cpu()[::48, :, 7, 0].reshape(-1)[30:], color='tab:green', label="STT Importance of Jus")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 7, 0].reshape(-1)[30:], color='tab:blue', label="TFT Importance of Jus")
#%%
np.mean(eval_label.squeeze().cpu().numpy() < stt_output[..., 4].squeeze().detach().cpu().numpy()) ### 0.924
np.mean(eval_label.squeeze().cpu().numpy() < output_no_sps[..., 4].squeeze().detach().cpu().numpy()) ### 0.778
np.mean(eval_label.squeeze().cpu().numpy() < output_parallel[..., 4].squeeze().detach().cpu().numpy()) ### 0.936

# np.mean(eval_label.squeeze().cpu().numpy() < tft_output[..., 4].squeeze().detach().cpu().numpy()) ### 0.870
# np.mean(eval_label.squeeze().cpu().numpy() < mqrnn_output[..., 4].squeeze().detach().cpu().numpy()) ### 0.870
np.mean(eval_label.squeeze().cpu().numpy() < deepar_output[..., 4].squeeze()) ### 0.722

#%%
import pytorch_model_summary as pms
pms.summary(stt2, eval_conti, eval_cate, eval_future, show_input=True, print_summary=True)
pms.summary(tft, eval_conti, eval_cate, eval_future, show_input=True, print_summary=True)
pms.summary(deepar, eval_conti, eval_cate, eval_future, show_input=True, print_summary=True)
pms.summary(mqrnn, eval_conti, eval_cate, eval_future, show_input=True, print_summary=True)

np.mean(eval_label.squeeze().cpu().numpy() < stt_output[..., 3].squeeze().detach().cpu().numpy()) ### 0.638
np.mean(eval_label.squeeze().cpu().numpy() < output_no_sps[..., 3].squeeze().detach().cpu().numpy()) ### 0.640
np.mean(eval_label.squeeze().cpu().numpy() < output_parallel[..., 3].squeeze().detach().cpu().numpy()) ### 0.894
np.mean(eval_label.squeeze().cpu().numpy() < deepar_output[..., 3].squeeze()) ### 0.721

np.mean(eval_label.squeeze().cpu().numpy() < stt_output[..., 2].squeeze().detach().cpu().numpy()) ### 0.623
np.mean(eval_label.squeeze().cpu().numpy() < output_no_sps[..., 2].squeeze().detach().cpu().numpy()) ### 0.640
np.mean(eval_label.squeeze().cpu().numpy() < output_parallel[..., 2].squeeze().detach().cpu().numpy()) ### 0.823
np.mean(eval_label.squeeze().cpu().numpy() < deepar_output[..., 2].squeeze()) ### 0.720
#%%
#%%
torch.maximum(0.9 * (eval_label.squeeze() - stt_output[..., 4].squeeze()), (1-0.9)*(stt_output[..., 4].squeeze() -eval_label.squeeze() )).mean() # 0.0031
torch.maximum(0.7 * (eval_label.squeeze() - stt_output[..., 3].squeeze()), (1-0.7)*(stt_output[..., 3].squeeze() -eval_label.squeeze() )).mean() # 0.0051
torch.maximum(0.5 * (eval_label.squeeze() - stt_output[..., 2].squeeze()), (1-0.5)*(stt_output[..., 2].squeeze() -eval_label.squeeze() )).mean() # 0.0059

#%%
torch.maximum(0.9 * (eval_label.squeeze() - output_no_sps[..., 4].squeeze()), (1-0.9)*(output_no_sps[..., 4].squeeze() -eval_label.squeeze() )).mean() # 0.0031
torch.maximum(0.7 * (eval_label.squeeze() - output_no_sps[..., 3].squeeze()), (1-0.7)*(output_no_sps[..., 3].squeeze() -eval_label.squeeze() )).mean() # 0.0051
torch.maximum(0.5 * (eval_label.squeeze() - output_no_sps[..., 2].squeeze()), (1-0.5)*(output_no_sps[..., 2].squeeze() -eval_label.squeeze() )).mean() # 0.0059
#%%
torch.maximum(0.9 * (eval_label.squeeze() - output_parallel[..., 4].squeeze()), (1-0.9)*(output_parallel[..., 4].squeeze() -eval_label.squeeze() )).mean() # 0.0031
torch.maximum(0.7 * (eval_label.squeeze() - output_parallel[..., 3].squeeze()), (1-0.7)*(output_parallel[..., 3].squeeze() -eval_label.squeeze() )).mean() # 0.0051
torch.maximum(0.5 * (eval_label.squeeze() - output_parallel[..., 2].squeeze()), (1-0.5)*(output_parallel[..., 2].squeeze() -eval_label.squeeze() )).mean() # 0.0059
#%%
torch.maximum(0.9 * (eval_label.squeeze() - tft_output[..., 4].squeeze()), (1-0.9)*(tft_output[..., 4].squeeze() -eval_label.squeeze() )).mean() # 0.0031
torch.maximum(0.7 * (eval_label.squeeze() - tft_output[..., 3].squeeze()), (1-0.7)*(tft_output[..., 3].squeeze() -eval_label.squeeze() )).mean() # 0.0051
torch.maximum(0.5 * (eval_label.squeeze() - tft_output[..., 2].squeeze()), (1-0.5)*(tft_output[..., 2].squeeze() -eval_label.squeeze() )).mean() # 0.0059
#%%
torch.maximum(0.9 * (eval_label.squeeze() - mqrnn_output[..., 4].squeeze()), (1-0.9)*(mqrnn_output[..., 4].squeeze() -eval_label.squeeze() )).mean() # 0.0030
torch.maximum(0.7 * (eval_label.squeeze() - mqrnn_output[..., 3].squeeze()), (1-0.7)*(mqrnn_output[..., 3].squeeze() -eval_label.squeeze() )).mean() # 0.0051
torch.maximum(0.5 * (eval_label.squeeze() - mqrnn_output[..., 2].squeeze()), (1-0.5)*(mqrnn_output[..., 2].squeeze() -eval_label.squeeze() )).mean() # 0.0053

torch.maximum(0.9 * (eval_label.squeeze() - torch.Tensor(deepar_output)[..., 4].squeeze()), (1-0.9)*(torch.Tensor(deepar_output)[..., 4].squeeze() -eval_label.squeeze() )).mean() # 0.0030
torch.maximum(0.7 * (eval_label.squeeze() - torch.Tensor(deepar_output)[..., 3].squeeze()), (1-0.7)*(torch.Tensor(deepar_output)[..., 3].squeeze() -eval_label.squeeze() )).mean() # 0.0051
torch.maximum(0.5 * (eval_label.squeeze() - torch.Tensor(deepar_output)[..., 2].squeeze()), (1-0.5)*(torch.Tensor(deepar_output)[..., 2].squeeze() -eval_label.squeeze() )).mean() # 0.0053

np.mean(eval_label.squeeze().cpu().numpy() > stt_output[..., 4].squeeze().detach().cpu().numpy())
np.mean(eval_label.squeeze().cpu().numpy() > tft_output[..., 4].squeeze().detach().cpu().numpy())
np.mean(eval_label.squeeze().cpu().numpy() > mqrnn_output[..., 4].squeeze().detach().cpu().numpy())
np.mean(eval_label.squeeze().cpu().numpy() < deepar_output[..., 2].squeeze())
#%%
ql = QuantileRisk(tau=12, quantile=[0.1, 0.3, 0.5, 0.7, 0.9], num_targets=1, device=device)

ql(eval_label, torch.Tensor(deepar_output))
ql(eval_label, mqrnn_output)
ql(eval_label, tft_output)
ql(eval_label, stt_output)
# ql(eval_label, stt_recalibration_output.unsqueeze(-2))
#%% 변수 중요도 시각화
eval_conti = torch.FloatTensor(eval_a[1500:1650])
eval_cate = torch.LongTensor(eval_b[1500:1650])
eval_future = torch.LongTensor(eval_c[1500:1650])
eval_label = torch.FloatTensor(eval_d[1500:1650])
eval_past_label = torch.FloatTensor(eval_e[1500:1650])
#%%
eval_conti = torch.FloatTensor(eval_a[:744, ...])
eval_cate = torch.LongTensor(eval_b[:744, ...])
eval_future = torch.LongTensor(eval_c[:744, ...])
eval_label = torch.FloatTensor(eval_d[:744, ...])
# eval_past_label = torch.FloatTensor(eval_e[3012:3732, ...])
#%%
stt2.eval()
stt_output, ssa_weight1, ssa_weight2, tsa_weight, dec_weights, fi1, fi2  = stt2(eval_conti, eval_cate, eval_future)

tft.eval()
confe_output = tft.confe(eval_conti) # (batch_size, seq_len, num_cv, d_embedding)
catfe_output = tft.catfe(eval_cate)  # (batch_size, seq_len, num_cate, d_embedding)
obs_feature = torch.cat([confe_output, catfe_output], axis=-2)  # (batch_size, seq_len, num_cv + num_cate, d_embedding)
x1, tft_vsn_output  = tft.vsn1(obs_feature) # (batch_size, seq_len, d_model)
#%%
mpl.rcParams["figure.dpi"] = 300
mpl_style(dark=False)

#%% P1 importance
ax = sns.lineplot(eval_conti.cpu()[::48, :, 0].squeeze().reshape(-1), label=r"Observation of $P_1$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 0, 0].reshape(-1), label=r"Importance of $P_1$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 0, 0].reshape(-1), label=r"Importance of $P_1$ (TFT)")
plt.xticks([0, 240, 480, 720], ['05-01', '05-10', '05-20', '05-30']) # ax.set_xticks([0, 240, 480]) 
# ax.set_xticklabels() # ax.set_xticklabels(['10-01', '10-10', '10-20']) #
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 1].squeeze().reshape(-1), label=r"Observation $P_2$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 1, 0].reshape(-1), label=r"Importance of $P_2$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 1, 0].reshape(-1), label=r"Importance of $P_2$ (TFT)")
plt.xticks([0, 240, 480, 720], ['05-01', '05-10', '05-20', '05-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 2].squeeze().reshape(-1), label=r"Observation of $P_3$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 2, 0].reshape(-1), label=r"Importance of $P_3$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 2, 0].reshape(-1), label=r"Importance of $P_3$ (TFT)")
plt.xticks([0, 240, 480, 720], ['05-01', '05-10', '05-20', '05-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 3].squeeze().reshape(-1), label=r"Observation of $WL(B_5)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 3, 0].reshape(-1), label=r"Importance of $WL(B_5)$ (InstaTran)")
# sns.lineplot(fi1.detach().cpu()[::48, :, 11, 0].reshape(-1), label="WL(JS)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 3, 0].reshape(-1),label=r"Importance of $WL(B_5)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['05-01', '05-10', '05-20', '05-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 4].squeeze().reshape(-1), label=r"Observation of $WL(D)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 4, 0].reshape(-1), label=r"Importance of $WL(D)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 4, 0].reshape(-1),label=r"Importance of $WL(D)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['05-01', '05-10', '05-20', '05-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 5].squeeze().reshape(-1), label=r"Observation of $IF(D)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 5, 0].reshape(-1), label=r"Importance of $IF(D)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 5, 0].reshape(-1),label=r"Importance of $IF(D)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['05-01', '05-10', '05-20', '05-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 6].squeeze().reshape(-1), label=r"Observation of $STR(D)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 6, 0].reshape(-1), label=r"Importance of $STR(D)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 6, 0].reshape(-1),label=r"Importance of $STR(D)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['05-01', '05-10', '05-20', '05-30'])
#%% JUS importance
ax = sns.lineplot(eval_conti.cpu()[::48, :, 7].squeeze().reshape(-1), label=r"Observation of $JUS$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 7, 0].reshape(-1), label=r"Importance of $JUS$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 7, 0].reshape(-1),label=r"Importance of $JUS$ (TFT)")
plt.xticks([0, 240, 480, 720], ['05-01', '05-10', '05-20', '05-30'])
#%% OF importance
ax = sns.lineplot(eval_conti.cpu()[::48, :, 8].squeeze().reshape(-1), label=r"Observation of $OF$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 8, 0].reshape(-1), label=r"Importance of $OF$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 8, 0].reshape(-1), label=r"Importance of $OF$ (TFT)")
plt.xticks([0, 240, 480, 720], ['05-01', '05-10', '05-20', '05-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 9].squeeze().reshape(-1), label=r"Observation of $WL(B_1)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 9, 0].reshape(-1), label=r"Importance of $WL(B_1)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 9, 0].reshape(-1),label=r"Importance of $WL(B_1)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['05-01', '05-10', '05-20', '05-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 10].squeeze().reshape(-1), label=r"Observation of $FL(B_1)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 10, 0].reshape(-1), label="Importance of $FL(B_1)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 10, 0].reshape(-1), label="Importance of $FL(B_1)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['05-01', '05-10', '05-20', '05-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 11].squeeze().reshape(-1), label=r"Observation of $WL(B_2)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 11, 0].reshape(-1),label="Importance of $WL(B_2)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 11, 0].reshape(-1), label="Importance of $WL(B_2)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['05-01', '05-10', '05-20', '05-30'])
#%% Hangang WL importance
ax = sns.lineplot(eval_conti.cpu()[::48, :, 12].squeeze().reshape(-1), label=r"Observation of $WL(B_3)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 12, 0].reshape(-1), label="Importance of $WL(B_3)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 12, 0].reshape(-1), label="Importance of $WL(B_3)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['05-01', '05-10', '05-20', '05-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 13].squeeze().reshape(-1), label=r"Observation of $FL(B_3)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 13, 0].reshape(-1), label="Importance of $FL(B_3)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 13, 0].reshape(-1), label="Importance of $FL(B_3)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['05-01', '05-10', '05-20', '05-30'])
#%% Hangju WL importance
ax = sns.lineplot(eval_conti.cpu()[::48, :, 14].squeeze().reshape(-1), label=r"Observation of $WL(B_4)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 14, 0].reshape(-1), label="Importance of $WL(B_4)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 14, 0].reshape(-1), label="Importance of $WL(B_4)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['05-01', '05-10', '05-20', '05-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 15].squeeze().reshape(-1), label=r"Observation of $FL(B_4)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 15, 0].reshape(-1), label="Importance of $FL(B_4)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 15, 0].reshape(-1), label="Importance of $FL(B_4)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['05-01', '05-10', '05-20', '05-30'])
#%%
#%% 6월 결과 plotting
eval_conti = torch.FloatTensor(eval_a[744:1464, ...])
eval_cate = torch.LongTensor(eval_b[744:1464, ...])
eval_future = torch.LongTensor(eval_c[744:1464, ...])
eval_label = torch.FloatTensor(eval_d[744:1464, ...])
# eval_past_label = torch.FloatTensor(eval_e[744:1464, ...])
# %%
stt2.eval()
stt_output, ssa_weight1, ssa_weight2, tsa_weight, dec_weights, fi1, fi2  = stt2(eval_conti, eval_cate, eval_future)

tft.eval()
confe_output = tft.confe(eval_conti) # (batch_size, seq_len, num_cv, d_embedding)
catfe_output = tft.catfe(eval_cate)  # (batch_size, seq_len, num_cate, d_embedding)
obs_feature = torch.cat([confe_output, catfe_output], axis=-2)  # (batch_size, seq_len, num_cv + num_cate, d_embedding)
x1, tft_vsn_output  = tft.vsn1(obs_feature) # (batch_size, seq_len, d_model)
#%% P1 importance
ax = sns.lineplot(eval_conti.cpu()[::48, :, 0].squeeze().reshape(-1), label=r"Observation of $P_1$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 0, 0].reshape(-1), label=r"Importance of $P_1$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 0, 0].reshape(-1), label=r"Importance of $P_1$ (TFT)")
plt.xticks([0, 240, 480, 720], ['06-01', '06-10', '06-20', '06-30']) # ax.set_xticks([0, 240, 480]) 
# ax.set_xticklabels() # ax.set_xticklabels(['10-01', '10-10', '10-20']) #
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 1].squeeze().reshape(-1), label=r"Observation $P_2$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 1, 0].reshape(-1), label=r"Importance of $P_2$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 1, 0].reshape(-1), label=r"Importance of $P_2$ (TFT)")
plt.xticks([0, 240, 480, 720], ['06-01', '06-10', '06-20', '06-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 2].squeeze().reshape(-1), label=r"Observation of $P_3$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 2, 0].reshape(-1), label=r"Importance of $P_3$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 2, 0].reshape(-1), label=r"Importance of $P_3$ (TFT)")
plt.xticks([0, 240, 480, 720], ['06-01', '06-10', '06-20', '06-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 3].squeeze().reshape(-1), label=r"Observation of $WL(B_5)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 3, 0].reshape(-1), label=r"Importance of $WL(B_5)$ (InstaTran)")
# sns.lineplot(fi1.detach().cpu()[::48, :, 11, 0].reshape(-1), label="WL(JS)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 3, 0].reshape(-1),label=r"Importance of $WL(B_5)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['06-01', '06-10', '06-20', '06-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 4].squeeze().reshape(-1), label=r"Observation of $WL(D)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 4, 0].reshape(-1), label=r"Importance of $WL(D)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 4, 0].reshape(-1),label=r"Importance of $WL(D)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['06-01', '06-10', '06-20', '06-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 5].squeeze().reshape(-1), label=r"Observation of $IF(D)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 5, 0].reshape(-1), label=r"Importance of $IF(D)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 5, 0].reshape(-1),label=r"Importance of $IF(D)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['06-01', '06-10', '06-20', '06-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 6].squeeze().reshape(-1), label=r"Observation of $STR(D)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 6, 0].reshape(-1), label=r"Importance of $STR(D)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 6, 0].reshape(-1),label=r"Importance of $STR(D)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['06-01', '06-10', '06-20', '06-30'])
#%% JUS importance
ax = sns.lineplot(eval_conti.cpu()[::48, :, 7].squeeze().reshape(-1), label=r"Observation of $JUS$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 7, 0].reshape(-1), label=r"Importance of $JUS$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 7, 0].reshape(-1),label=r"Importance of $JUS$ (TFT)")
plt.xticks([0, 240, 480, 720], ['06-01', '06-10', '06-20', '06-30'])
#%% OF importance
ax = sns.lineplot(eval_conti.cpu()[::48, :, 8].squeeze().reshape(-1), label=r"Observation of $OF$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 8, 0].reshape(-1), label=r"Importance of $OF$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 8, 0].reshape(-1), label=r"Importance of $OF$ (TFT)")
plt.xticks([0, 240, 480, 720], ['06-01', '06-10', '06-20', '06-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 9].squeeze().reshape(-1), label=r"Observation of $WL(B_1)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 9, 0].reshape(-1), label=r"Importance of $WL(B_1)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 9, 0].reshape(-1),label=r"Importance of $WL(B_1)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['06-01', '06-10', '06-20', '06-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 10].squeeze().reshape(-1), label=r"Observation of $FL(B_1)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 10, 0].reshape(-1), label="Importance of $FL(B_1)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 10, 0].reshape(-1), label="Importance of $FL(B_1)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['06-01', '06-10', '06-20', '06-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 11].squeeze().reshape(-1), label=r"Observation of $WL(B_2)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 11, 0].reshape(-1),label="Importance of $WL(B_2)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 11, 0].reshape(-1), label="Importance of $WL(B_2)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['06-01', '06-10', '06-20', '06-30'])
#%% Hangang WL importance
ax = sns.lineplot(eval_conti.cpu()[::48, :, 12].squeeze().reshape(-1), label=r"Observation of $WL(B_3)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 12, 0].reshape(-1), label="Importance of $WL(B_3)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 12, 0].reshape(-1), label="Importance of $WL(B_3)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['06-01', '06-10', '06-20', '06-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 13].squeeze().reshape(-1), label=r"Observation of $FL(B_3)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 13, 0].reshape(-1), label="Importance of $FL(B_3)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 13, 0].reshape(-1), label="Importance of $FL(B_3)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['06-01', '06-10', '06-20', '06-30'])
#%% Hangju WL importance
ax = sns.lineplot(eval_conti.cpu()[::48, :, 14].squeeze().reshape(-1), label=r"Observation of $WL(B_4)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 14, 0].reshape(-1), label="Importance of $WL(B_4)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 14, 0].reshape(-1), label="Importance of $WL(B_4)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['06-01', '06-10', '06-20', '06-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 15].squeeze().reshape(-1), label=r"Observation of $FL(B_4)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 15, 0].reshape(-1), label="Importance of $FL(B_4)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 15, 0].reshape(-1), label="Importance of $FL(B_4)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['06-01', '06-10', '06-20', '06-30'])
# %%
#%% 7월 결과
eval_conti = torch.FloatTensor(eval_a[1464:2268, ...])
eval_cate = torch.LongTensor(eval_b[1464:2268, ...])
eval_future = torch.LongTensor(eval_c[1464:2268, ...])
eval_label = torch.FloatTensor(eval_d[1464:2268, ...])
#%%
stt2.eval()
stt_output, ssa_weight1, ssa_weight2, tsa_weight, dec_weights, fi1, fi2  = stt2(eval_conti, eval_cate, eval_future)

tft.eval()
confe_output = tft.confe(eval_conti) # (batch_size, seq_len, num_cv, d_embedding)
catfe_output = tft.catfe(eval_cate)  # (batch_size, seq_len, num_cate, d_embedding)
obs_feature = torch.cat([confe_output, catfe_output], axis=-2)  # (batch_size, seq_len, num_cv + num_cate, d_embedding)
x1, tft_vsn_output  = tft.vsn1(obs_feature) # (batch_size, seq_len, d_model)
#%% P1 importance
ax = sns.lineplot(eval_conti.cpu()[::48, :, 0].squeeze().reshape(-1), label=r"Observation of $P_1$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 0, 0].reshape(-1), label=r"Importance of $P_1$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 0, 0].reshape(-1), label=r"Importance of $P_1$ (TFT)")
plt.xticks([0, 240, 480, 720], ['07-01', '07-10', '07-20', '07-30']) # ax.set_xticks([0, 240, 480]) 
# ax.set_xticklabels() # ax.set_xticklabels(['10-01', '10-10', '10-20']) #
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 1].squeeze().reshape(-1), label=r"Observation $P_2$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 1, 0].reshape(-1), label=r"Importance of $P_2$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 1, 0].reshape(-1), label=r"Importance of $P_2$ (TFT)")
plt.xticks([0, 240, 480, 720], ['07-01', '07-10', '07-20', '07-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 2].squeeze().reshape(-1), label=r"Observation of $P_3$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 2, 0].reshape(-1), label=r"Importance of $P_3$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 2, 0].reshape(-1), label=r"Importance of $P_3$ (TFT)")
plt.xticks([0, 240, 480, 720], ['07-01', '07-10', '07-20', '07-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 3].squeeze().reshape(-1), label=r"Observation of $WL(B_5)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 3, 0].reshape(-1), label=r"Importance of $WL(B_5)$ (InstaTran)")
# sns.lineplot(fi1.detach().cpu()[::48, :, 11, 0].reshape(-1), label="WL(JS)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 3, 0].reshape(-1),label=r"Importance of $WL(B_5)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['07-01', '07-10', '07-20', '07-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 4].squeeze().reshape(-1), label=r"Observation of $WL(D)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 4, 0].reshape(-1), label=r"Importance of $WL(D)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 4, 0].reshape(-1),label=r"Importance of $WL(D)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['07-01', '07-10', '07-20', '07-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 5].squeeze().reshape(-1), label=r"Observation of $IF(D)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 5, 0].reshape(-1), label=r"Importance of $IF(D)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 5, 0].reshape(-1),label=r"Importance of $IF(D)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['07-01', '07-10', '07-20', '07-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 6].squeeze().reshape(-1), label=r"Observation of $STR(D)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 6, 0].reshape(-1), label=r"Importance of $STR(D)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 6, 0].reshape(-1),label=r"Importance of $STR(D)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['07-01', '07-10', '07-20', '07-30'])
#%% JUS importance
ax = sns.lineplot(eval_conti.cpu()[::48, :, 7].squeeze().reshape(-1), label=r"Observation of $JUS$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 7, 0].reshape(-1), label=r"Importance of $JUS$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 7, 0].reshape(-1),label=r"Importance of $JUS$ (TFT)")
plt.xticks([0, 240, 480, 720], ['07-01', '07-10', '07-20', '07-30'])
#%% OF importance
ax = sns.lineplot(eval_conti.cpu()[::48, :, 8].squeeze().reshape(-1), label=r"Observation of $OF$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 8, 0].reshape(-1), label=r"Importance of $OF$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 8, 0].reshape(-1), label=r"Importance of $OF$ (TFT)")
plt.xticks([0, 240, 480, 720], ['07-01', '07-10', '07-20', '07-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 9].squeeze().reshape(-1), label=r"Observation of $WL(B_1)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 9, 0].reshape(-1), label=r"Importance of $WL(B_1)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 9, 0].reshape(-1),label=r"Importance of $WL(B_1)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['07-01', '07-10', '07-20', '07-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 10].squeeze().reshape(-1), label=r"Observation of $FL(B_1)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 10, 0].reshape(-1), label="Importance of $FL(B_1)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 10, 0].reshape(-1), label="Importance of $FL(B_1)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['07-01', '07-10', '07-20', '07-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 11].squeeze().reshape(-1), label=r"Observation of $WL(B_2)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 11, 0].reshape(-1),label="Importance of $WL(B_2)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 11, 0].reshape(-1), label="Importance of $WL(B_2)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['07-01', '07-10', '07-20', '07-30'])
#%% Hangang WL importance
ax = sns.lineplot(eval_conti.cpu()[::48, :, 12].squeeze().reshape(-1), label=r"Observation of $WL(B_3)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 12, 0].reshape(-1), label="Importance of $WL(B_3)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 12, 0].reshape(-1), label="Importance of $WL(B_3)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['07-01', '07-10', '07-20', '07-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 13].squeeze().reshape(-1), label=r"Observation of $FL(B_3)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 13, 0].reshape(-1), label="Importance of $FL(B_3)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 13, 0].reshape(-1), label="Importance of $FL(B_3)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['07-01', '07-10', '07-20', '07-30'])
#%% Hangju WL importance
ax = sns.lineplot(eval_conti.cpu()[::48, :, 14].squeeze().reshape(-1), label=r"Observation of $WL(B_4)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 14, 0].reshape(-1), label="Importance of $WL(B_4)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 14, 0].reshape(-1), label="Importance of $WL(B_4)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['07-01', '07-10', '07-20', '07-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 15].squeeze().reshape(-1), label=r"Observation of $FL(B_4)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 15, 0].reshape(-1), label="Importance of $FL(B_4)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 15, 0].reshape(-1), label="Importance of $FL(B_4)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['07-01', '07-10', '07-20', '07-30'])
#%% 8월
eval_conti = torch.FloatTensor(eval_a[2268:3012, ...])
eval_cate = torch.LongTensor(eval_b[2268:3012, ...])
eval_future = torch.LongTensor(eval_c[2268:3012, ...])
eval_label = torch.FloatTensor(eval_d[2268:3012, ...])
#%%
stt2.eval()
stt_output, ssa_weight1, ssa_weight2, tsa_weight, dec_weights, fi1, fi2  = stt2(eval_conti, eval_cate, eval_future)

tft.eval()
confe_output = tft.confe(eval_conti) # (batch_size, seq_len, num_cv, d_embedding)
catfe_output = tft.catfe(eval_cate)  # (batch_size, seq_len, num_cate, d_embedding)
obs_feature = torch.cat([confe_output, catfe_output], axis=-2)  # (batch_size, seq_len, num_cv + num_cate, d_embedding)
x1, tft_vsn_output  = tft.vsn1(obs_feature) # (batch_size, seq_len, d_model)
#%% P1 importance
ax = sns.lineplot(eval_conti.cpu()[::48, :, 0].squeeze().reshape(-1), label=r"Observation of $P_1$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 0, 0].reshape(-1), label=r"Importance of $P_1$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 0, 0].reshape(-1), label=r"Importance of $P_1$ (TFT)")
plt.xticks([0, 240, 480, 720], ['08-01', '08-10', '08-20', '08-30']) # ax.set_xticks([0, 240, 480]) 
# ax.set_xticklabels() # ax.set_xticklabels(['10-01', '10-10', '10-20']) #
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 1].squeeze().reshape(-1), label=r"Observation $P_2$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 1, 0].reshape(-1), label=r"Importance of $P_2$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 1, 0].reshape(-1), label=r"Importance of $P_2$ (TFT)")
plt.xticks([0, 240, 480, 720], ['08-01', '08-10', '08-20', '08-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 2].squeeze().reshape(-1), label=r"Observation of $P_3$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 2, 0].reshape(-1), label=r"Importance of $P_3$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 2, 0].reshape(-1), label=r"Importance of $P_3$ (TFT)")
plt.xticks([0, 240, 480, 720], ['08-01', '08-10', '08-20', '08-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 3].squeeze().reshape(-1), label=r"Observation of $WL(B_5)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 3, 0].reshape(-1), label=r"Importance of $WL(B_5)$ (InstaTran)")
# sns.lineplot(fi1.detach().cpu()[::48, :, 11, 0].reshape(-1), label="WL(JS)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 3, 0].reshape(-1),label=r"Importance of $WL(B_5)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['08-01', '08-10', '08-20', '08-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 4].squeeze().reshape(-1), label=r"Observation of $WL(D)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 4, 0].reshape(-1), label=r"Importance of $WL(D)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 4, 0].reshape(-1),label=r"Importance of $WL(D)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['08-01', '08-10', '08-20', '08-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 5].squeeze().reshape(-1), label=r"Observation of $IF(D)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 5, 0].reshape(-1), label=r"Importance of $IF(D)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 5, 0].reshape(-1),label=r"Importance of $IF(D)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['08-01', '08-10', '08-20', '08-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 6].squeeze().reshape(-1), label=r"Observation of $STR(D)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 6, 0].reshape(-1), label=r"Importance of $STR(D)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 6, 0].reshape(-1),label=r"Importance of $STR(D)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['08-01', '08-10', '08-20', '08-30'])
#%% JUS importance
ax = sns.lineplot(eval_conti.cpu()[::48, :, 7].squeeze().reshape(-1), label=r"Observation of $JUS$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 7, 0].reshape(-1), label=r"Importance of $JUS$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 7, 0].reshape(-1),label=r"Importance of $JUS$ (TFT)")
plt.xticks([0, 240, 480, 720], ['08-01', '08-10', '08-20', '08-30'])
#%% OF importance
ax = sns.lineplot(eval_conti.cpu()[::48, :, 8].squeeze().reshape(-1), label=r"Observation of $OF$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 8, 0].reshape(-1), label=r"Importance of $OF$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 8, 0].reshape(-1), label=r"Importance of $OF$ (TFT)")
plt.xticks([0, 240, 480, 720], ['08-01', '08-10', '08-20', '08-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 9].squeeze().reshape(-1), label=r"Observation of $WL(B_1)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 9, 0].reshape(-1), label=r"Importance of $WL(B_1)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 9, 0].reshape(-1),label=r"Importance of $WL(B_1)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['08-01', '08-10', '08-20', '08-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 10].squeeze().reshape(-1), label=r"Observation of $FL(B_1)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 10, 0].reshape(-1), label="Importance of $FL(B_1)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 10, 0].reshape(-1), label="Importance of $FL(B_1)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['08-01', '08-10', '08-20', '08-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 11].squeeze().reshape(-1), label=r"Observation of $WL(B_2)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 11, 0].reshape(-1),label="Importance of $WL(B_2)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 11, 0].reshape(-1), label="Importance of $WL(B_2)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['08-01', '08-10', '08-20', '08-30'])
#%% Hangang WL importance
ax = sns.lineplot(eval_conti.cpu()[::48, :, 12].squeeze().reshape(-1), label=r"Observation of $WL(B_3)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 12, 0].reshape(-1), label="Importance of $WL(B_3)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 12, 0].reshape(-1), label="Importance of $WL(B_3)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['08-01', '08-10', '08-20', '08-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 13].squeeze().reshape(-1), label=r"Observation of $FL(B_3)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 13, 0].reshape(-1), label="Importance of $FL(B_3)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 13, 0].reshape(-1), label="Importance of $FL(B_3)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['08-01', '08-10', '08-20', '08-30'])
#%% Hangju WL importance
ax = sns.lineplot(eval_conti.cpu()[::48, :, 14].squeeze().reshape(-1), label=r"Observation of $WL(B_4)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 14, 0].reshape(-1), label="Importance of $WL(B_4)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 14, 0].reshape(-1), label="Importance of $WL(B_4)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['08-01', '08-10', '08-20', '08-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 15].squeeze().reshape(-1), label=r"Observation of $FL(B_4)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 15, 0].reshape(-1), label="Importance of $FL(B_4)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 15, 0].reshape(-1), label="Importance of $FL(B_4)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['08-01', '08-10', '08-20', '08-30'])
#%% 9월
eval_conti = torch.FloatTensor(eval_a[3012:3732, ...])
eval_cate = torch.LongTensor(eval_b[3012:3732, ...])
eval_future = torch.LongTensor(eval_c[3012:3732, ...])
eval_label = torch.FloatTensor(eval_d[3012:3732, ...])
#%%
stt2.eval()
stt_output, ssa_weight1, ssa_weight2, tsa_weight, dec_weights, fi1, fi2  = stt2(eval_conti, eval_cate, eval_future)

tft.eval()
confe_output = tft.confe(eval_conti) # (batch_size, seq_len, num_cv, d_embedding)
catfe_output = tft.catfe(eval_cate)  # (batch_size, seq_len, num_cate, d_embedding)
obs_feature = torch.cat([confe_output, catfe_output], axis=-2)  # (batch_size, seq_len, num_cv + num_cate, d_embedding)
x1, tft_vsn_output  = tft.vsn1(obs_feature) # (batch_size, seq_len, d_model)
#%% P1 importance
ax = sns.lineplot(eval_conti.cpu()[::48, :, 0].squeeze().reshape(-1), label=r"Observation of $P_1$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 0, 0].reshape(-1), label=r"Importance of $P_1$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 0, 0].reshape(-1), label=r"Importance of $P_1$ (TFT)")
plt.xticks([0, 240, 480, 720], ['09-01', '09-10', '09-20', '09-30']) # ax.set_xticks([0, 240, 480]) 
# ax.set_xticklabels() # ax.set_xticklabels(['10-01', '10-10', '10-20']) #
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 1].squeeze().reshape(-1), label=r"Observation $P_2$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 1, 0].reshape(-1), label=r"Importance of $P_2$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 1, 0].reshape(-1), label=r"Importance of $P_2$ (TFT)")
plt.xticks([0, 240, 480, 720], ['09-01', '09-10', '09-20', '09-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 2].squeeze().reshape(-1), label=r"Observation of $P_3$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 2, 0].reshape(-1), label=r"Importance of $P_3$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 2, 0].reshape(-1), label=r"Importance of $P_3$ (TFT)")
plt.xticks([0, 240, 480, 720], ['09-01', '09-10', '09-20', '09-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 3].squeeze().reshape(-1), label=r"Observation of $WL(B_5)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 3, 0].reshape(-1), label=r"Importance of $WL(B_5)$ (InstaTran)")
# sns.lineplot(fi1.detach().cpu()[::48, :, 11, 0].reshape(-1), label="WL(JS)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 3, 0].reshape(-1),label=r"Importance of $WL(B_5)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['09-01', '09-10', '09-20', '09-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 4].squeeze().reshape(-1), label=r"Observation of $WL(D)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 4, 0].reshape(-1), label=r"Importance of $WL(D)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 4, 0].reshape(-1),label=r"Importance of $WL(D)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['09-01', '09-10', '09-20', '09-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 5].squeeze().reshape(-1), label=r"Observation of $IF(D)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 5, 0].reshape(-1), label=r"Importance of $IF(D)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 5, 0].reshape(-1),label=r"Importance of $IF(D)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['09-01', '09-10', '09-20', '09-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 6].squeeze().reshape(-1), label=r"Observation of $STR(D)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 6, 0].reshape(-1), label=r"Importance of $STR(D)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 6, 0].reshape(-1),label=r"Importance of $STR(D)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['09-01', '09-10', '09-20', '09-30'])
#%% JUS importance
ax = sns.lineplot(eval_conti.cpu()[::48, :, 7].squeeze().reshape(-1), label=r"Observation of $JUS$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 7, 0].reshape(-1), label=r"Importance of $JUS$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 7, 0].reshape(-1),label=r"Importance of $JUS$ (TFT)")
plt.xticks([0, 240, 480, 720], ['09-01', '09-10', '09-20', '09-30'])
#%% OF importance
ax = sns.lineplot(eval_conti.cpu()[::48, :, 8].squeeze().reshape(-1), label=r"Observation of $OF$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 8, 0].reshape(-1), label=r"Importance of $OF$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 8, 0].reshape(-1), label=r"Importance of $OF$ (TFT)")
plt.xticks([0, 240, 480, 720], ['09-01', '09-10', '09-20', '09-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 9].squeeze().reshape(-1), label=r"Observation of $WL(B_1)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 9, 0].reshape(-1), label=r"Importance of $WL(B_1)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 9, 0].reshape(-1),label=r"Importance of $WL(B_1)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['09-01', '09-10', '09-20', '09-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 10].squeeze().reshape(-1), label=r"Observation of $FL(B_1)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 10, 0].reshape(-1), label="Importance of $FL(B_1)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 10, 0].reshape(-1), label="Importance of $FL(B_1)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['09-01', '09-10', '09-20', '09-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 11].squeeze().reshape(-1), label=r"Observation of $WL(B_2)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 11, 0].reshape(-1),label="Importance of $WL(B_2)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 11, 0].reshape(-1), label="Importance of $WL(B_2)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['09-01', '09-10', '09-20', '09-30'])
#%% Hangang WL importance
ax = sns.lineplot(eval_conti.cpu()[::48, :, 12].squeeze().reshape(-1), label=r"Observation of $WL(B_3)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 12, 0].reshape(-1), label="Importance of $WL(B_3)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 12, 0].reshape(-1), label="Importance of $WL(B_3)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['09-01', '09-10', '09-20', '09-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 13].squeeze().reshape(-1), label=r"Observation of $FL(B_3)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 13, 0].reshape(-1), label="Importance of $FL(B_3)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 13, 0].reshape(-1), label="Importance of $FL(B_3)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['09-01', '09-10', '09-20', '09-30'])
#%% Hangju WL importance
ax = sns.lineplot(eval_conti.cpu()[::48, :, 14].squeeze().reshape(-1), label=r"Observation of $WL(B_4)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 14, 0].reshape(-1), label="Importance of $WL(B_4)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 14, 0].reshape(-1), label="Importance of $WL(B_4)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['09-01', '09-10', '09-20', '09-30'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 15].squeeze().reshape(-1), label=r"Observation of $FL(B_4)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 15, 0].reshape(-1), label="Importance of $FL(B_4)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 15, 0].reshape(-1), label="Importance of $FL(B_4)$ (TFT)")
plt.xticks([0, 240, 480, 720], ['09-01', '09-10', '09-20', '09-30'])
# %%
#%% 10월
eval_conti = torch.FloatTensor(eval_a[3732:, ...])
eval_cate = torch.LongTensor(eval_b[3732:, ...])
eval_future = torch.LongTensor(eval_c[3732:, ...])
eval_label = torch.FloatTensor(eval_d[3732:, ...])
#%%
stt2.eval()
stt_output, ssa_weight1, ssa_weight2, tsa_weight, dec_weights, fi1, fi2  = stt2(eval_conti, eval_cate, eval_future)

tft.eval()
confe_output = tft.confe(eval_conti) # (batch_size, seq_len, num_cv, d_embedding)
catfe_output = tft.catfe(eval_cate)  # (batch_size, seq_len, num_cate, d_embedding)
obs_feature = torch.cat([confe_output, catfe_output], axis=-2)  # (batch_size, seq_len, num_cv + num_cate, d_embedding)
x1, tft_vsn_output  = tft.vsn1(obs_feature) # (batch_size, seq_len, d_model)
#%% P1 importance
ax = sns.lineplot(eval_conti.cpu()[::48, :, 0].squeeze().reshape(-1), label=r"Observation of $P_1$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 0, 0].reshape(-1), label=r"Importance of $P_1$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 0, 0].reshape(-1), label=r"Importance of $P_1$ (TFT)")
plt.xticks([0, 240, 480], ['10-01', '10-10', '10-20']) # ax.set_xticks([0, 240, 480]) 
# ax.set_xticklabels() # ax.set_xticklabels(['10-01', '10-10', '10-20']) #
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 1].squeeze().reshape(-1), label=r"Observation $P_2$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 1, 0].reshape(-1), label=r"Importance of $P_2$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 1, 0].reshape(-1), label=r"Importance of $P_2$ (TFT)")
plt.xticks([0, 240, 480], ['10-01', '10-10', '10-20'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 2].squeeze().reshape(-1), label=r"Observation of $P_3$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 2, 0].reshape(-1), label=r"Importance of $P_3$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 2, 0].reshape(-1), label=r"Importance of $P_3$ (TFT)")
plt.xticks([0, 240, 480], ['10-01', '10-10', '10-20'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 3].squeeze().reshape(-1), label=r"Observation of $WL(B_5)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 3, 0].reshape(-1), label=r"Importance of $WL(B_5)$ (InstaTran)")
# sns.lineplot(fi1.detach().cpu()[::48, :, 11, 0].reshape(-1), label="WL(JS)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 3, 0].reshape(-1),label=r"Importance of $WL(B_5)$ (TFT)")
plt.xticks([0, 240, 480], ['10-01', '10-10', '10-20'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 4].squeeze().reshape(-1), label=r"Observation of $WL(D)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 4, 0].reshape(-1), label=r"Importance of $WL(D)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 4, 0].reshape(-1),label=r"Importance of $WL(D)$ (TFT)")
plt.xticks([0, 240, 480], ['10-01', '10-10', '10-20'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 5].squeeze().reshape(-1), label=r"Observation of $IF(D)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 5, 0].reshape(-1), label=r"Importance of $IF(D)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 5, 0].reshape(-1),label=r"Importance of $IF(D)$ (TFT)")
plt.xticks([0, 240, 480], ['10-01', '10-10', '10-20'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 6].squeeze().reshape(-1), label=r"Observation of $STR(D)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 6, 0].reshape(-1), label=r"Importance of $STR(D)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 6, 0].reshape(-1),label=r"Importance of $STR(D)$ (TFT)")
plt.xticks([0, 240, 480], ['10-01', '10-10', '10-20'])
#%% JUS importance
ax = sns.lineplot(eval_conti.cpu()[::48, :, 7].squeeze().reshape(-1), label=r"Observation of $JUS$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 7, 0].reshape(-1), label=r"Importance of $JUS$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 7, 0].reshape(-1),label=r"Importance of $JUS$ (TFT)")
plt.xticks([0, 240, 480], ['10-01', '10-10', '10-20'])
#%% OF importance
ax = sns.lineplot(eval_conti.cpu()[::48, :, 8].squeeze().reshape(-1), label=r"Observation of $OF$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 8, 0].reshape(-1), label=r"Importance of $OF$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 8, 0].reshape(-1), label=r"Importance of $OF$ (TFT)")
plt.xticks([0, 240, 480], ['10-01', '10-10', '10-20'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 9].squeeze().reshape(-1), label=r"Observation of $WL(B_1)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 9, 0].reshape(-1), label=r"Importance of $WL(B_1)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 9, 0].reshape(-1),label=r"Importance of $WL(B_1)$ (TFT)")
plt.xticks([0, 240, 480], ['10-01', '10-10', '10-20'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 10].squeeze().reshape(-1), label=r"Observation of $FL(B_1)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 10, 0].reshape(-1), label="Importance of $FL(B_1)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 10, 0].reshape(-1), label="Importance of $FL(B_1)$ (TFT)")
plt.xticks([0, 240, 480], ['10-01', '10-10', '10-20'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 11].squeeze().reshape(-1), label=r"Observation of $WL(B_2)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 11, 0].reshape(-1),label="Importance of $WL(B_2)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 11, 0].reshape(-1), label="Importance of $WL(B_2)$ (TFT)")
plt.xticks([0, 240, 480], ['10-01', '10-10', '10-20'])
#%% Hangang WL importance
ax = sns.lineplot(eval_conti.cpu()[::48, :, 12].squeeze().reshape(-1), label=r"Observation of $WL(B_3)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 12, 0].reshape(-1), label="Importance of $WL(B_3)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 12, 0].reshape(-1), label="Importance of $WL(B_3)$ (TFT)")
plt.xticks([0, 240, 480], ['10-01', '10-10', '10-20'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 13].squeeze().reshape(-1), label=r"Observation of $FL(B_3)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 13, 0].reshape(-1), label="Importance of $FL(B_3)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 13, 0].reshape(-1), label="Importance of $FL(B_3)$ (TFT)")
plt.xticks([0, 240, 480], ['10-01', '10-10', '10-20'])
#%% Hangju WL importance
ax = sns.lineplot(eval_conti.cpu()[::48, :, 14].squeeze().reshape(-1), label=r"Observation of $WL(B_4)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 14, 0].reshape(-1), label="Importance of $WL(B_4)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 14, 0].reshape(-1), label="Importance of $WL(B_4)$ (TFT)")
plt.xticks([0, 240, 480], ['10-01', '10-10', '10-20'])
#%%
ax = sns.lineplot(eval_conti.cpu()[::48, :, 15].squeeze().reshape(-1), label=r"Observation of $FL(B_4)$").set(xlabel="Time points")
sns.lineplot(fi1.detach().cpu()[::48, :, 15, 0].reshape(-1), label="Importance of $FL(B_4)$ (InstaTran)")
sns.lineplot(tft_vsn_output.detach().cpu()[::48, :, 15, 0].reshape(-1), label="Importance of $FL(B_4)$ (TFT)")
plt.xticks([0, 240, 480], ['10-01', '10-10', '10-20'])
# %%
eval_conti.shape
fi1.shape
feature_idx = 7
importance_mat = np.zeros((624, 671))
for i in range(624):
    importance_mat[i, i:i+48] = fi1.detach().cpu()[i, :, feature_idx, 0]


tft_importance_mat = np.zeros((624, 671))
for i in range(624):
    tft_importance_mat[i, i:i+48] = tft_vsn_output.detach().cpu()[i, :, feature_idx, 0]

ax = sns.lineplot(eval_conti.cpu()[::48, :, feature_idx].squeeze().reshape(-1), label=r"Observation of $P_1$").set(xlabel="Time points")
sns.lineplot(np.nanmean(np.where(importance_mat==0.0, np.nan, importance_mat), axis=0), linestyle='--', label=r"Importance of $P_1$ (InstaTran)")
sns.lineplot(np.nanmean(np.where(tft_importance_mat==0.0, np.nan, tft_importance_mat), axis=0), linestyle=':', label=r"Importance of $P_1$ (TFT)")
# %%
eval_conti = torch.FloatTensor(eval_a)
eval_cate = torch.LongTensor(eval_b)
eval_future = torch.LongTensor(eval_c)
eval_label = torch.FloatTensor(eval_d)
eval_past_label = torch.FloatTensor(eval_e)

# stt_output, ssa_weight1, ssa_weight2, tsa_weight, dec_weights, fi1, fi2  = stt(eval_conti, eval_cate, eval_future)

stt2.eval()
stt_output, ssa_weight1, ssa_weight2, tsa_weight, dec_weights, fi1, fi2  = stt2(eval_conti, eval_cate, eval_future)
tft.eval()
tft_output, decoder_weights = tft(eval_conti, eval_cate, eval_future)
#%%
tau = 49
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, tau, :48], 0.9, axis=0))
sns.lineplot(dec_weights.detach().numpy()[:, tau, :48].mean(axis=0))
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, tau, :48], 0.1, axis=0))

tau = 54
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, tau, :48], 0.9, axis=0))
sns.lineplot(dec_weights.detach().numpy()[:, tau, :48].mean(axis=0))
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, tau, :48], 0.1, axis=0))

tau = 59
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, tau, :48], 0.9, axis=0))
sns.lineplot(dec_weights.detach().numpy()[:, tau, :48].mean(axis=0))
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, tau, :48], 0.1, axis=0))
#%%
tau = 49
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, tau, :48], 0.9, axis=0))
sns.lineplot(decoder_weights.detach().numpy()[:, tau, :48].mean(axis=0))
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, tau, :48], 0.1, axis=0))

tau = 54
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, tau, :48], 0.9, axis=0))
sns.lineplot(decoder_weights.detach().numpy()[:, tau, :48].mean(axis=0))
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, tau, :48], 0.1, axis=0))

tau = 59
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, tau, :48], 0.9, axis=0))
sns.lineplot(decoder_weights.detach().numpy()[:, tau, :48].mean(axis=0))
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, tau, :48], 0.1, axis=0))
# %%
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 48, :48], 0.5, axis=0), label=r"$\tau = 1$")
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 49, :48], 0.5, axis=0), label=r"$\tau = 2$")
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 50, :48], 0.5, axis=0), label=r"$\tau = 3$")
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 51, :48], 0.5, axis=0), label=r"$\tau = 4$")
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 52, :48], 0.5, axis=0), label=r"$\tau = 5$")
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 53, :48], 0.5, axis=0), label=r"$\tau = 6$")
#%%
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 54, :48], 0.5, axis=0), label=r"$\tau = 7$")
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 55, :48], 0.5, axis=0), label=r"$\tau = 8$")
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 56, :48], 0.5, axis=0), label=r"$\tau = 9$")
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 57, :48], 0.5, axis=0), label=r"$\tau = 10$")
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 58, :48], 0.5, axis=0), label=r"$\tau = 11$")
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 59, :48], 0.5, axis=0), label=r"$\tau = 12$")
#%%
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, 48, :48], 0.5, axis=0), label=r"$\tau = 1$")
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, 49, :48], 0.5, axis=0), label=r"$\tau = 2$")
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, 50, :48], 0.5, axis=0), label=r"$\tau = 3$")
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, 51, :48], 0.5, axis=0), label=r"$\tau = 4$")
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, 52, :48], 0.5, axis=0), label=r"$\tau = 5$")
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, 53, :48], 0.5, axis=0), label=r"$\tau = 6$")
#%%
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, 54, :48], 0.5, axis=0), label=r"$\tau = 7$")
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, 55, :48], 0.5, axis=0), label=r"$\tau = 8$")
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, 56, :48], 0.5, axis=0), label=r"$\tau = 9$")
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, 57, :48], 0.5, axis=0), label=r"$\tau = 10$")
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, 58, :48], 0.5, axis=0), label=r"$\tau = 11$")
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, 59, :48], 0.5, axis=0), label=r"$\tau = 12$")
#%%
mpl.rcParams["figure.dpi"] = 200
mpl_style(dark=False)
SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
#%%
g = sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 48, ], 0.5, axis=0), label=r"$k = 1$")
g.set_ylim(0, 0.029)
g.set_xticks([0, 12, 24, 36, 47, 59], ["-47", "-35", "-23", "-11", "0", "12"])
g.set_xlabel("Time points")
g.set_ylabel("Median of attention weights")
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 49, ], 0.5, axis=0), label=r"$k = 2$")
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 50, ], 0.5, axis=0), label=r"$k = 3$")
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 51, ], 0.5, axis=0), label=r"$k = 4$")
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 52, ], 0.5, axis=0), label=r"$k = 5$")
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 53, ], 0.5, axis=0), label=r"$k = 6$")
g.axvline(47,  linestyle='--', linewidth=2, color='k')
# g.axvline(11,  linestyle=':', linewidth=2, color='k')
# g.axvline(23,  linestyle=':', linewidth=2, color='k')
# g.axvline(35,  linestyle=':', linewidth=2, color='k')
xstart = 24
ystart = 0.026
g.annotate("",
            xy=(xstart, ystart),
            xytext=(xstart+12, ystart),
            va="center",
            ha="center",
            arrowprops=dict(color='black', arrowstyle="<->"))
g.annotate("Half-daily interval", xy=(xstart-4, ystart+0.002), xytext=(xstart-0.7, ystart+0.002), color='black')
g.annotate("(12 hours)", xy=(xstart, ystart+0.0005), xytext=(xstart+2.4, ystart+0.0005), color='black')
#%%
g = sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 54, :], 0.5, axis=0), label=r"$k = 7$")
g.set_ylim(0, 0.029)
g.set_xticks([0, 12, 24, 36, 47, 59], ["-47", "-35", "-23", "-11", "0", "12"])
g.set_xlabel("Time points")
g.set_ylabel("Median of attention weights")
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 55, :], 0.5, axis=0), label=r"$k = 8$")
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 56, :], 0.5, axis=0), label=r"$k = 9$")
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 57, :], 0.5, axis=0), label=r"$k = 10$")
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 58, :], 0.5, axis=0), label=r"$k = 11$")
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 59, :], 0.5, axis=0), label=r"$k = 12$")
g.axvline(47,  linestyle='--', linewidth=2, color='k')
xstart = 28
ystart = 0.026
g.annotate("",
            xy=(xstart, ystart),
            xytext=(xstart+12, ystart),
            va="center",
            ha="center",
            arrowprops=dict(color='black', arrowstyle="<->"))
g.annotate("Half-daily interval", xy=(xstart-4, ystart+0.002), xytext=(xstart-0.7, ystart+0.002), color='black')
g.annotate("(12 hours)", xy=(xstart, ystart+0.0005), xytext=(xstart+2.4, ystart+0.0005), color='black')
#%%
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 48, :], 0.5, axis=0), label=r"$k = 1$")
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 54, :], 0.5, axis=0), label=r"$k = 7$")
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 59, :], 0.5, axis=0), label=r"$k = 12$")
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 57, :], 0.5, axis=0), label=r"$k = 10$")
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 58, :], 0.5, axis=0), label=r"$k = 11$")
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 59, :], 0.5, axis=0), label=r"$k = 12$")
#%%
g = sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, 48, ], 0.5, axis=0), label=r"$k = 1$")
g.set_ylim(0, 0.029)
g.set_xticks([0, 12, 24, 36, 47, 59], ["-47", "-35", "-23", "-11", "0", "12"])
g.set_xlabel("Time points")
g.set_ylabel("Median of attention weights")
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, 49, ], 0.5, axis=0), label=r"$k = 2$")
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, 50, ], 0.5, axis=0), label=r"$k = 3$")
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, 51, ], 0.5, axis=0), label=r"$k = 4$")
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, 52, ], 0.5, axis=0), label=r"$k = 5$")
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, 53, ], 0.5, axis=0), label=r"$k = 6$")
g.axvline(47,  linestyle='--', linewidth=2, color='k')
xstart = 24
ystart = 0.024
g.annotate("",
            xy=(xstart, ystart),
            xytext=(xstart+12, ystart),
            va="center",
            ha="center",
            arrowprops=dict(color='black', arrowstyle="<->"))
g.annotate("Half-daily interval", xy=(xstart-4, ystart+0.002), xytext=(xstart-0.7, ystart+0.002), color='black')
g.annotate("(12 hours)", xy=(xstart, ystart+0.0005), xytext=(xstart+2.4, ystart+0.0005), color='black')
#%%
g = sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, 54, :], 0.5, axis=0), label=r"$k = 7$")
g.set_ylim(0, 0.029)
g.set_xticks([0, 12, 24, 36, 47, 59], ["-47", "-35", "-23", "-11", "0", "12"])
g.set_xlabel("Time points")
g.set_ylabel("Median of attention weights")
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, 55, :], 0.5, axis=0), label=r"$k = 8$")
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, 56, :], 0.5, axis=0), label=r"$k = 9$")
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, 57, :], 0.5, axis=0), label=r"$k = 10$")
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, 58, :], 0.5, axis=0), label=r"$k = 11$")
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, 59, :], 0.5, axis=0), label=r"$k = 12$")
g.axvline(47,  linestyle='--', linewidth=2, color='k')
xstart = 25
ystart = 0.023
g.annotate("",
            xy=(xstart, ystart),
            xytext=(xstart+12, ystart),
            va="center",
            ha="center",
            arrowprops=dict(color='black', arrowstyle="<->"))
g.annotate("Half-daily interval", xy=(xstart-4, ystart+0.002), xytext=(xstart-0.7, ystart+0.002), color='black')
g.annotate("(12 hours)", xy=(xstart, ystart+0.0005), xytext=(xstart+2.4, ystart+0.0005), color='black')
# %%
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 49, :48], 0.5, axis=0))
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 50, :48], 0.5, axis=0))
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 51, :48], 0.5, axis=0))

#%%
g = sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 48, :], 0.1, axis=0), label=r"10%")
g.set_xticks([0, 12, 24, 36, 47, 59], ["-47", "-35", "-23", "-11", "0", "12"])
g.set_xlabel("Time points")
g.set_ylabel("Attention weights")
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 48, :], 0.5, axis=0), label=r"50%")
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 48, :], 0.9, axis=0), label=r"90%")
g.axvline(47,  linestyle='--', linewidth=2, color='k')
g.axvline(11,  linestyle=':', linewidth=2, color='k')
g.axvline(23,  linestyle=':', linewidth=2, color='k')
g.axvline(35,  linestyle=':', linewidth=2, color='k')
g.annotate("",
            xy=(11, 0.045),
            xytext=(23, 0.045),
            va="center",
            ha="center",
            arrowprops=dict(color='black', arrowstyle="<->"))
g.annotate("12 hours", xy=(11, 0.046), xytext=(13.4, 0.046), color='black')
#%%
g = sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 54, :], 0.1, axis=0), label=r"10%")
g.set_xticks([0, 12, 24, 36, 47, 59], ["-47", "-35", "-23", "-11", "0", "12"])
g.set_xlabel("Time points")
g.set_ylabel("Attention weights")
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 54, :], 0.5, axis=0), label=r"50%")
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 54, :], 0.9, axis=0), label=r"90%")
g.axvline(47,  linestyle='--', linewidth=2, color='k')
g.axvline(17,  linestyle=':', linewidth=2, color='k')
g.axvline(29,  linestyle=':', linewidth=2, color='k')
g.axvline(41,  linestyle=':', linewidth=2, color='k')
g.annotate("",
            xy=(17, 0.041),
            xytext=(29, 0.041),
            va="center",
            ha="center",
            arrowprops=dict(color='black', arrowstyle="<->"))
g.annotate("12 hours", xy=(11, 0.042), xytext=(19.2, 0.0415), color='black')
# %%
dec_weights.detach().numpy()[:, 48, 48]

torch.triu(torch.ones([60, 60], dtype=torch.bool), diagonal=1)[47, 47:]
np.quantile(dec_weights.detach().numpy()[:, 53, :], 0.5, axis=0)[15]
np.quantile(dec_weights.detach().numpy()[:, 53, :], 0.5, axis=0)[27]
np.quantile(dec_weights.detach().numpy()[:, 53, :], 0.5, axis=0)[40]
#%%
tau = 49
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, tau, :48], 0.9, axis=0))
sns.lineplot(decoder_weights.detach().numpy()[:, tau, :48].mean(axis=0))
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, tau, :48], 0.1, axis=0))

tau = 54
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, tau, :48], 0.9, axis=0))
sns.lineplot(decoder_weights.detach().numpy()[:, tau, :48].mean(axis=0))
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, tau, :48], 0.1, axis=0))

tau = 59
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, tau, :48], 0.9, axis=0))
sns.lineplot(decoder_weights.detach().numpy()[:, tau, :48].mean(axis=0))
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, tau, :48], 0.1, axis=0))
#%%
g = sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, 48, ], 0.5, axis=0), label=r"$k = 1$")
g.set_xticks([0, 12, 24, 36, 47, 59], ["-47", "-35", "-23", "-11", "0", "12"])
g.set_xlabel("Time points")
g.set_ylabel("Median of Attention Weights")
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, 49, ], 0.5, axis=0), label=r"$k = 2$")
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, 50, ], 0.5, axis=0), label=r"$k = 3$")
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, 51, ], 0.5, axis=0), label=r"$k = 4$")
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, 52, ], 0.5, axis=0), label=r"$k = 5$")
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, 53, ], 0.5, axis=0), label=r"$k = 6$")
# g.axvline(47,  linestyle='--', linewidth=2, color='k')
# g.axvline(11,  linestyle=':', linewidth=2, color='k')
# g.axvline(23,  linestyle=':', linewidth=2, color='k')
# g.axvline(35,  linestyle=':', linewidth=2, color='k')
# g.annotate("",
#             xy=(7, 0.018),
#             xytext=(19, 0.018),
#             va="center",
#             ha="center",
#             arrowprops=dict(color='black', arrowstyle="<->"))
# g.annotate("12 hours", xy=(7, 0.019), xytext=(9.4, 0.019), color='black')
# %%
g = sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, 54, ], 0.5, axis=0), label=r"$k = 7$")
g.set_xticks([0, 12, 24, 36, 47, 59], ["-47", "-35", "-23", "-11", "0", "12"])
g.set_xlabel("Time points")
g.set_ylabel("Median of Attention Weights")
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, 55, ], 0.5, axis=0), label=r"$k = 8$")
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, 56, ], 0.5, axis=0), label=r"$k = 9$")
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, 57, ], 0.5, axis=0), label=r"$k = 10$")
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, 58, ], 0.5, axis=0), label=r"$k = 11$")
sns.lineplot(np.quantile(decoder_weights.detach().numpy()[:, 59, ], 0.5, axis=0), label=r"$k = 12$")
# g.axvline(47,  linestyle='--', linewidth=2, color='k')
# g.axvline(11,  linestyle=':', linewidth=2, color='k')
# g.axvline(23,  linestyle=':', linewidth=2, color='k')
# g.axvline(35,  linestyle=':', linewidth=2, color='k')
# g.annotate("",
#             xy=(7, 0.018),
#             xytext=(19, 0.018),
#             va="center",
#             ha="center",
#             arrowprops=dict(color='black', arrowstyle="<->"))
# g.annotate("12 hours", xy=(7, 0.019), xytext=(9.4, 0.019), color='black')
# %%
stt2_without.eval()
output_no_sps, ssa_weight1_no_sps, ssa_weight2_no_sps, tsa_weight_no_sps, dec_weights_no_sps, fi1_no_sps, fi2_no_sps = stt2_without(eval_conti, eval_cate, eval_future)
#%%
g = sns.lineplot(np.quantile(dec_weights_no_sps.detach().numpy()[:, 48, ], 0.5, axis=0), label=r"$k = 1$")
g.set_xticks([0, 12, 24, 36, 47, 59], ["-47", "-35", "-23", "-11", "0", "12"])
g.set_xlabel("Time points")
g.set_ylabel("Attention weights")
sns.lineplot(np.quantile(dec_weights_no_sps.detach().numpy()[:, 49, ], 0.5, axis=0), label=r"$k = 2$")
sns.lineplot(np.quantile(dec_weights_no_sps.detach().numpy()[:, 50, ], 0.5, axis=0), label=r"$k = 3$")
sns.lineplot(np.quantile(dec_weights_no_sps.detach().numpy()[:, 51, ], 0.5, axis=0), label=r"$k = 4$")
sns.lineplot(np.quantile(dec_weights_no_sps.detach().numpy()[:, 52, ], 0.5, axis=0), label=r"$k = 5$")
sns.lineplot(np.quantile(dec_weights_no_sps.detach().numpy()[:, 53, ], 0.5, axis=0), label=r"$k = 6$")
# g.axvline(47,  linestyle='--', linewidth=2, color='k')
# g.axvline(11,  linestyle=':', linewidth=2, color='k')
# g.axvline(23,  linestyle=':', linewidth=2, color='k')
# g.axvline(35,  linestyle=':', linewidth=2, color='k')
# g.annotate("",
#             xy=(11, 0.045),
#             xytext=(23, 0.045),
#             va="center",
#             ha="center",
#             arrowprops=dict(color='black', arrowstyle="<->"))
# g.annotate("12 hours", xy=(11, 0.046), xytext=(13.4, 0.046), color='black')
# %%
g = sns.lineplot(np.quantile(dec_weights_no_sps.detach().numpy()[:, 48, ], 0.5, axis=0), label=r"$k = 1$")
g.set_ylim(0, 0.029)
g.set_xticks([0, 12, 24, 36, 47, 59], ["-47", "-35", "-23", "-11", "0", "12"])
g.set_xlabel("Time points")
g.set_ylabel("Median of attention weights")
sns.lineplot(np.quantile(dec_weights_no_sps.detach().numpy()[:, 49, ], 0.5, axis=0), label=r"$k = 2$")
sns.lineplot(np.quantile(dec_weights_no_sps.detach().numpy()[:, 50, ], 0.5, axis=0), label=r"$k = 3$")
sns.lineplot(np.quantile(dec_weights_no_sps.detach().numpy()[:, 51, ], 0.5, axis=0), label=r"$k = 4$")
sns.lineplot(np.quantile(dec_weights_no_sps.detach().numpy()[:, 52, ], 0.5, axis=0), label=r"$k = 5$")
sns.lineplot(np.quantile(dec_weights_no_sps.detach().numpy()[:, 53, ], 0.5, axis=0), label=r"$k = 6$")
#%%
stt_parallel.eval()
output_parallel, ssa_weight1_parallel, ssa_weight2_parallel, tsa_weight_parallel, dec_weights_parallel, fi1_parallel, fi2_parallel = stt_parallel(eval_conti, eval_cate, eval_future)
# %%
g = sns.lineplot(np.quantile(dec_weights_no_sps.detach().numpy()[:, 54, ], 0.5, axis=0), label=r"$k = 7$")
g.set_ylim(0, 0.029)
g.set_xticks([0, 12, 24, 36, 47, 59], ["-47", "-35", "-23", "-11", "0", "12"])
g.set_xlabel("Time points")
g.set_ylabel("Median of attention weights")
sns.lineplot(np.quantile(dec_weights_no_sps.detach().numpy()[:, 55, ], 0.5, axis=0), label=r"$k = 8$")
sns.lineplot(np.quantile(dec_weights_no_sps.detach().numpy()[:, 56, ], 0.5, axis=0), label=r"$k = 9$")
sns.lineplot(np.quantile(dec_weights_no_sps.detach().numpy()[:, 57, ], 0.5, axis=0), label=r"$k = 10$")
sns.lineplot(np.quantile(dec_weights_no_sps.detach().numpy()[:, 58, ], 0.5, axis=0), label=r"$k = 11$")
sns.lineplot(np.quantile(dec_weights_no_sps.detach().numpy()[:, 59, ], 0.5, axis=0), label=r"$k = 12$")
#%%
g = sns.lineplot(np.quantile(dec_weights_parallel.detach().numpy()[:, 48, ], 0.5, axis=0), label=r"$k = 1$")
g.set_ylim(0, 0.033)
g.set_xticks([0, 12, 24, 36, 47, 59], ["-47", "-35", "-23", "-11", "0", "12"])
g.set_xlabel("Time points")
g.set_ylabel("Median of attention weights")
sns.lineplot(np.quantile(dec_weights_parallel.detach().numpy()[:, 49, ], 0.5, axis=0), label=r"$k = 2$")
sns.lineplot(np.quantile(dec_weights_parallel.detach().numpy()[:, 50, ], 0.5, axis=0), label=r"$k = 3$")
sns.lineplot(np.quantile(dec_weights_parallel.detach().numpy()[:, 51, ], 0.5, axis=0), label=r"$k = 4$")
sns.lineplot(np.quantile(dec_weights_parallel.detach().numpy()[:, 52, ], 0.5, axis=0), label=r"$k = 5$")
sns.lineplot(np.quantile(dec_weights_parallel.detach().numpy()[:, 53, ], 0.5, axis=0), label=r"$k = 6$")
# %%
g = sns.lineplot(np.quantile(dec_weights_parallel.detach().numpy()[:, 54, ], 0.5, axis=0), label=r"$k = 7$")
g.set_ylim(0, 0.033)
g.set_xticks([0, 12, 24, 36, 47, 59], ["-47", "-35", "-23", "-11", "0", "12"])
g.set_xlabel("Time points")
g.set_ylabel("Median of attention weights")
sns.lineplot(np.quantile(dec_weights_parallel.detach().numpy()[:, 55, ], 0.5, axis=0), label=r"$k = 8$")
sns.lineplot(np.quantile(dec_weights_parallel.detach().numpy()[:, 56, ], 0.5, axis=0), label=r"$k = 9$")
sns.lineplot(np.quantile(dec_weights_parallel.detach().numpy()[:, 57, ], 0.5, axis=0), label=r"$k = 10$")
sns.lineplot(np.quantile(dec_weights_parallel.detach().numpy()[:, 58, ], 0.5, axis=0), label=r"$k = 11$")
sns.lineplot(np.quantile(dec_weights_parallel.detach().numpy()[:, 59, ], 0.5, axis=0), label=r"$k = 12$")
#%%
output, ssa_weight1, ssa_weight2, tsa_weight, dec_weights, fi1, fi2 = stt(eval_conti, eval_cate, eval_future)
#%%
g = sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 48, ], 0.5, axis=0), label=r"$k = 1$")
g.set_ylim(0, 0.029)
g.set_xticks([0, 12, 24, 36, 47, 59], ["-47", "-35", "-23", "-11", "0", "12"])
g.set_xlabel("Time points")
g.set_ylabel("Median of attention weights")
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 49, ], 0.5, axis=0), label=r"$k = 2$")
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 50, ], 0.5, axis=0), label=r"$k = 3$")
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 51, ], 0.5, axis=0), label=r"$k = 4$")
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 52, ], 0.5, axis=0), label=r"$k = 5$")
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 53, ], 0.5, axis=0), label=r"$k = 6$")
g.axvline(47,  linestyle='--', linewidth=2, color='k')
# g.axvline(11,  linestyle=':', linewidth=2, color='k')
# g.axvline(23,  linestyle=':', linewidth=2, color='k')
# g.axvline(35,  linestyle=':', linewidth=2, color='k')

#%%
g = sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 54, ], 0.5, axis=0), label=r"$k = 7$")
g.set_ylim(0, 0.033)
g.set_xticks([0, 12, 24, 36, 47, 59], ["-47", "-35", "-23", "-11", "0", "12"])
g.set_xlabel("Time points")
g.set_ylabel("Median of attention weights")
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 55, ], 0.5, axis=0), label=r"$k = 8$")
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 56, ], 0.5, axis=0), label=r"$k = 9$")
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 57, ], 0.5, axis=0), label=r"$k = 10$")
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 58, ], 0.5, axis=0), label=r"$k = 11$")
sns.lineplot(np.quantile(dec_weights.detach().numpy()[:, 59, ], 0.5, axis=0), label=r"$k = 12$")
#%%
output, ssa_weight1, ssa_weight2, tsa_weight, dec_weights, fi1, fi2 = stt(eval_conti, eval_cate, eval_future)

# %% Rainy day performance 
eval_conti = torch.FloatTensor(eval_a)
eval_cate = torch.LongTensor(eval_b)
eval_future = torch.LongTensor(eval_c)
eval_label = torch.FloatTensor(eval_d)
# eval_past_label = torch.FloatTensor(eval_e)
eval_conti.shape

eval_conti.cpu()[::48, :, 0].shape
plt.hist(eval_conti.cpu()[::48, :, 0].numpy().squeeze().reshape(-1))
eval_conti.cpu()[::48, :, 0].numpy().squeeze()
#%%
np.where(eval_conti.cpu()[:, :, 0].numpy().squeeze() > 0.1)

rainy_x_idx, rainy_y_idx = np.where((eval_conti.cpu()[:, :, 0].numpy().squeeze() > 0.1) & \
    (eval_conti.cpu()[:, :, 1].numpy().squeeze() > 0.1) & \
    (eval_conti.cpu()[:, :, 2].numpy().squeeze() > 0.1))
# %%
rainy_x_idx = np.unique(rainy_x_idx)

rainy_eval_conti = eval_conti[-rainy_x_idx, ... ]
rainy_eval_cate = eval_cate[-rainy_x_idx, ... ]
rainy_eval_future = eval_future[-rainy_x_idx, ... ]
rainy_eval_label = eval_label[-rainy_x_idx, ...]
#%%
stt2.eval()
rainy_stt_output, ssa_weight1, ssa_weight2, tsa_weight, dec_weights, fi1, fi2  = stt2(rainy_eval_conti,
                                                                                rainy_eval_cate,
                                                                                rainy_eval_future)
#%%
tft.eval()
rainy_tft_output, decoder_weights = tft(rainy_eval_conti, rainy_eval_cate, rainy_eval_future)
#%%
torch.maximum(0.9 * (rainy_eval_label.squeeze() - rainy_stt_output[..., 4].squeeze()), (1-0.9)*(rainy_stt_output[..., 4].squeeze() -rainy_eval_label.squeeze() )).mean() # 0.0031
torch.maximum(0.7 * (rainy_eval_label.squeeze() - rainy_stt_output[..., 3].squeeze()), (1-0.7)*(rainy_stt_output[..., 3].squeeze() -rainy_eval_label.squeeze() )).mean() # 0.0051
torch.maximum(0.5 * (rainy_eval_label.squeeze() - rainy_stt_output[..., 2].squeeze()), (1-0.5)*(rainy_stt_output[..., 2].squeeze() -rainy_eval_label.squeeze() )).mean() # 0.0059
#%%
torch.maximum(0.9 * (rainy_eval_label.squeeze() - rainy_tft_output[..., 4].squeeze()), (1-0.9)*(rainy_tft_output[..., 4].squeeze() -rainy_eval_label.squeeze() )).mean() # 0.0031
torch.maximum(0.7 * (rainy_eval_label.squeeze() - rainy_tft_output[..., 3].squeeze()), (1-0.7)*(rainy_tft_output[..., 3].squeeze() -rainy_eval_label.squeeze() )).mean() # 0.0051
torch.maximum(0.5 * (rainy_eval_label.squeeze() - rainy_tft_output[..., 2].squeeze()), (1-0.5)*(rainy_tft_output[..., 2].squeeze() -rainy_eval_label.squeeze() )).mean() # 0.0059