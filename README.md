# InstaTran

## Requirements

`Python 3.9.16`

### Using conda 
`conda create -n instatran python=3.9.16`

### Change directory
`cd instatran`

### 
` pip install -r requirements.txt`

## Dataset

### Download
Download data from one of the following links and unpack it into './'.
- [Han River Dataset](https://dacon.io/competitions/official/235949/data) (permission required)

Or you can use US lakes dataset in `./data/us_lakes/` without downloading.


## With US lakes dataset:
```
cd ./src
```

## Train InstaTran

```
python us_lakes_instatran.py
```

## Train neural-net-based benchmark models
```
python us_lakes_mqrnn.py
python us_lakes_deepar.py
python us_lakes_ding.py
python us_lakes_deng.py
python us_lakes_tft.py
```

## Train neural-net-based benchmark models
```
python us_lakes_mqrnn.py
python us_lakes_deepar.py
python us_lakes_ding.py
python us_lakes_deng.py
python us_lakes_tft.py
```

## Train classical benchmark models
```
python us_lakes_ml_benchmark.py
```
Or

use jupyter notebook `us_lakes_ml_benchmark.ipynb`


## If you download `Han River Dataset`:

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

### Saved InstaTran

```
/assets/instatran.pth
```

## Train neural-net-based benchmark models

```
python benchmark.py --model DeepAR --n_layer 3 
```

### Hyperparameters

- `model` : benchmark models (DeepAR, MQRnn, TFT)
- `n_layer` : the number of LSTM layers

### Saved benchmark Models

```
/assets/deepar.pth
/assets/mqrnn.pth
/assets/tft.pth
```

## Train classical benchmark models

Use jupyter notebook `classic_benchmark.ipynb`

## Distribution-shift Scenario

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
python dist_shift3.py --model Ding --year 2016
python dist_shift3.py --model Deng --year 2016
```

