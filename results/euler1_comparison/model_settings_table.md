# Euler-1 Model Settings

This table describes the current cleaned Euler-1 reduced-data surrogate comparison.

## Shared Training Setting

| Setting | Value |
|---|---:|
| Train / validation / test samples | 411 / 52 / 55 |
| Random seeds | 12345, 23456, 34567 |
| Maximum epochs | 100 |
| Early-stopping patience | 20 epochs |
| Batch size | 8 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | MSE |
| Gradient clipping | 10.0 |
| Model-selection metric | Validation RMSE |
| Input channels | 7 |
| Input-channel contents | Euler-1 known, strain rate, temperature, pressure, x grid, y grid, normalized step offset |
| Output channel | Euler-1 predicted field |
| Step-offset normalization bounds | 9 to 67 training steps |

## Model Architecture Setting

| Model | Role | Input channels | Main setting | Output activation | Trainable parameters |
|---|---|---:|---|---|---:|
| Persistence | No-training baseline | 7 | Returns the known Euler-1 channel unchanged | None | 0 |
| ResNet CNN | Local convolutional baseline | 7 | Width 32; 4 residual blocks; 3x3 convolutions; GELU activations | Tanh | 85,313 |
| FNO | Fourier neural operator surrogate | 7 | Width 16; 4 spectral blocks; 12 Fourier modes in each spatial direction; GELU activations | Tanh | 593,345 |

## Current Test Metrics

Mean +/- standard deviation over 3 random seeds.

| Model | Test RMSE | Test relative L2 | Circular RMSE (deg) | Circular MAE (deg) |
|---|---:|---:|---:|---:|
| Persistence | 0.29755 +/- 0 | 0.464883 +/- 0 | 32.8587 +/- 0 | 9.80081 +/- 0 |
| ResNet CNN | 0.176992 +/- 0.0010153 | 0.276527 +/- 0.00158628 | 27.3299 +/- 0.198451 | 11.9517 +/- 0.194865 |
| FNO | 0.174774 +/- 0.00434005 | 0.273062 +/- 0.00678076 | 26.8936 +/- 1.09549 | 12.1977 +/- 0.843617 |
