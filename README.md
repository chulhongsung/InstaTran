# instatran_IJF

```
cd ./src
```

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

