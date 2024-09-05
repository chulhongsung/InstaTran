# InstaTran

This repository contains the pre-release version of the dataset and Python code used in the paper "Interpretable Water Level Forecaster with Spatiotemporal Causal Attention Mechanisms". For detailed information about our model and implementation, please refer to the paper.
> Interpretable Water Level Forecaster with Spatiotemporal Causal Attention Mechanisms,  *S. Hong, Y. Choi, and J-J. Jeon*, *preprint*.

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

## Dataset

### Download
Download data from one of the following links and unpack it into `./data/raw`.
- [Han River Dataset](https://dacon.io/competitions/official/235949/data) (permission required)

### Preprocess of Han River Dataset  
Run jupyter notebook `preprocess.ipynb`

### US Lake dataset
Or you can use US lakes dataset in `./data/us_lakes` without permission.

## With US lakes dataset:
Change directory 
```
cd src
```

## Train and evaluate InstaTran

Run jupyter notebook `us_lakes_InstaTran.ipynb`

## Train and evaluate neural-net-based benchmark models

Run jupyter notebook `us_lakes_{model}.ipynb`
- `d_emb` : MQRnn, DeepAR, TFT, STALSTM, HSDSTM


## Train classical benchmark models (EPS, Theta, ARIMA, LightGBM)

Run jupyter notebook `us_lakes_ml_benchmark.ipynb`


## If you download `Han River Dataset`:
Note: This training process can be time-consuming. To save the time, the trained model  are saved in the `/assets/{model}.pth`.

## Train InstaTran

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

## Train neural-net-based benchmark models

```
python benchmark.py --model DeepAR --n_layer 3 
```

### Hyperparameters

- `model` : benchmark models (DeepAR, MQRnn, TFT)
- `n_layer` : the number of LSTM layers

## Train domain-specific benchmark models

Use jupyter notebook `HSTSTM.ipynb` and `STA-LSTM.ipynb`

### Saved benchmark Models

```
/assets/{model}.pth
```

## Train classical benchmark models
Run jupyter notebook `classic_benchmark.ipynb`

## Evaluate 
Run jupyter notebook `eval.ipynb`


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
python dist_shift2.py --model DeepAR --year 2016
python dist_shift2.py --model MQRnn --year 2016
python dist_shift2.py --model TFT --year 2016
```

### Train STA-LSTM, HSDSTM

```
python dist_shift3.py --model STALSTM --year 2016
python dist_shift3.py --model HSDSTM --year 2016
```

## Evaluation

Run jupyter notebook `eval_ds.ipynb`
