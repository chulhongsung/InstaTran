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

## Train InstaTran

```
cd ./src
```


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


## Train Benchmark models

```
python benchmark.py --model DeepAR --n_layer 3 
```

### Hyperparameters

- `model` : benchmark models (DeepAR, MQRnn, TFT)
- `n_layer` : the number of LSTM layers

### Saved Benchmark Models

```
/assets/deepar.pth
/assets/mqrnn.pth
/assets/tft.pth
```

