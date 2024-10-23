# InstaTran

This repository contains the pre-release version of the dataset and Python code used in the paper "Interpretable Water Level Forecaster with Spatiotemporal Causal Attention Mechanisms". For detailed information about our model and implementation, please refer to the paper.
> Interpretable Water Level Forecaster with Spatiotemporal Causal Attention Mechanisms, *preprint*.

## Requirements

`Python 3.9.16`

### Using conda 
`conda create -n {env_name} python=3.9.16`

`conda activate {env_name}`

### Clone repository
`git clone https://github.com/chulhongsung/InstaTran.git`

### Change directory
`cd InstaTran`

### 
`pip install -r requirements.txt`

> Note: If you encounter errors while installing LightGBM, you can remove`lightgbm==4.2.0` from the `requirements.txt` and install `ligthgbm` separately. 

## Folders
`src`: includes source codes

`data/raw`: a directory where to downlaod and save the Han River datasets (water_data, rf_data)

`data/us_lakes`: includes US lakes datasets

`asset`: includes outputs of numerical experiments

## Dataset

###  Han River Dataset

Raw datasets (water level and rainfall datasets) are obtained from [Open API](https://www.hrfco.go.kr/web/openapiPage/openApi.do).
- `./data/raw`: raw datasets (water_data (water levels from 2012 to 2022), rf_data (rainfall from 2012 to 2022))

### Preprocess of Han River Dataset  
Run jupyter notebook `preprocess.ipynb` to generate dataset (csv files) for training and evaluation with raw datasets.

#### outputs
- `df_merged.csv`
- `df_merged_ds_{year}.csv`
- `df_train_total.csv`
- `df_train_total_ds_{year}.csv`
- `df_test_total.csv`
- `df_test_total_ds_{year}.csv`

### US Lake dataset
Or you can use US lakes dataset in `./data/us_lakes`

## With US lakes dataset:
Change directory 
```
cd src
```

### Train and evaluate InstaTran

Run jupyter notebook `us_lakes_InstaTran.ipynb`

### Train and evaluate neural-net-based benchmark models

Run jupyter notebook `us_lakes_{model}.ipynb`
- `model` : MQRnn, DeepAR, TFT, STALSTM, HSDSTM

### Train classical benchmark models (EPS, Theta, ARIMA, LightGBM)

Run jupyter notebook `us_lakes_ml_benchmark.ipynb`


> Time (minutes) on various settings

| Models    | A6000 GPU (Ubuntu) | M2 Max CPU (Mac) | Intel i5-1340P CPU (Window) | 
| ----------| ------------------ | ---------------- | --------------------------- | 
| DeepAR    | 3 | 3  | 5 |
| MQRnn     | 5 | 6  | 10 | 
| STALSTM   | 1 | 1  | 1 | 
| HSDSTM    | 20 | 200  | 15 | 
| TFT       | 45 | 40  | 196 | 
| InstaTran | 50 | 114 | 147 |
| ML methods| 8 | 5 | 10 |


## `Han River Dataset`:
> Note: This training process can be time-consuming. To save the time, the trained model  are saved in the `/assets/{model}.pth`.

### Train InstaTran

```
python main.py 
```

### Hyperparameters

- `d_emb` : dimension of embedding vector
- `d_model` : dimension of hidden unit
- `bs` : batch size
- `epochs` : the number of epochs
- `lr` : learning rate

### Saved InstaTran directory

```
/assets/InstaTran.pth
```

### Ablation study 
```
python ablation.py 
```

#### Saved models

- `InstaTran_wo_sps.pth` (Without M_S) 
- `InstaTran_parallel.pth` (Parallel Attention)
- `InstaTran_w_tft_decoder.pth` (With TFT Decoder)
- `InstaTran_wo_M_S.pth` (Appendix E)

### Train neural-net-based benchmark models

```
python benchmark.py --model DeepAR --n_layer 3 --d_emb 3 --d_model 30
python benchmark.py --model MQRnn --n_layer 3 --d_emb 1 --d_model 5
python benchmark.py --model TFT --d_emb 5 --d_model 30
```

### Hyperparameters

- `model` : benchmark models (DeepAR, MQRnn, TFT)
- `n_layer` : the number of LSTM layers

### Train domain-specific benchmark models

Use jupyter notebook `HSTSTM.ipynb` and `STA-LSTM.ipynb`

### Saved benchmark Models

```
/assets/{model}.pth
```

### Train classical benchmark models
Run jupyter notebook `classic_benchmark.ipynb`

### Evaluate 
Run jupyter notebook `eval.ipynb`

> Time (minutes) on various settings

| Models    | A6000 GPU (Ubuntu) | M2 Max CPU (Mac) | Intel i5-1340P CPU (Window) | 
| ----------| ------------------ | ---------------- | --------------------------- | 
| DeepAR    | 6 | 21 | 33 |
| MQRnn     | 3 | 6 | 13 | 
| STALSTM   | 1 | 3  | 4 | 
| HSDSTM    | 3 |  76 | 33 | 
| TFT       | 33 |152 | 350 | 
| InstaTran | 22 | 158 | 236 |
| ML methods| 15 | 7 |  8  |


## Distribution-shift Scenario
Note: This training process can be time-consuming. To save the time, the trained model are saved in the `/assets/ds/ds_{model}_{year}.pth`.
- `model` : models (DeepAR, MQRnn, TFT, STALSTM, HSDSTM, instatran)
- `n_layer` : the number of LSTM layers


For example, if you train models with dataset in 2016, the parameter `year` is set as `2016`

### Train InstaTran
```
python dist_shift.py --year 2016
```

### Train DeepAR, MQRnn, TFT 

```
python dist_shift2.py --model DeepAR --year 2016 --d_model 30 --d_emb 3
python dist_shift2.py --model MQRnn --year 2016 --d_model 5 --d_emb 1
python dist_shift2.py --model TFT --year 2016 --d_model 30 --d_emb 5
```

### Train STA-LSTM, HSDSTM

```
python dist_shift3.py --model STALSTM --year 2016
python dist_shift3.py --model HSDSTM --year 2016
```

### Train and evaluate classical benchmark models

Run jupyter notebook `classic_benchmark_dist_shift.ipynb`

> Time (minutes) for 6-years datasets on various settings

| Models    | A6000 GPU (Ubuntu) | M2 Max CPU (Mac) | Intel i5-1340P CPU (Window) | 
| ----------| ------------------ | ---------------- | --------------------------- | 
| DeepAR    | 4 | 8 | 13 |
| MQRnn     | 3 | 3 | 7 | 
| STALSTM   | 4 |  6 | 7 | 
| HSDSTM    | 10 | 102 | 43 | 
| TFT       | 16 | 52 | 120 | 
| InstaTran | 36 | 141  | 220 |
| ML methods| 30 | 13 | 18 |



### Evaluation

Run jupyter notebook `eval_ds.ipynb`

## Reproducing Tables and Figures
> Note: run saved models by authors in `assets` folder because the results depend on the device you utilzed (CPU or GPU).

### `eval.ipynb`
- Figure 4 
- Figure 5 
- Figure 6 
- Figure E.10 
- Figure 7
- Figure D.9

- Table 2
- Table 3
- Table 5 (Deep learning based models)

### `classic_benchmark.ipynb`
- Table 5 (Classic models)

#### `eval_ds.ipynb`
- Table 6 (Deep learning based models)
- Figure 8

#### `classic_benchmark_dist_shift.ipynb`
- Table 6 (Classic models)

