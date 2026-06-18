# Euler-1 Reduced-Data Benchmark

This is a reduced Euler-1 comparison, not a full benchmark of the manuscript's grain-size plus three-Euler-angle microscale surrogate.

## Clean Split

- Train: 411 samples
- Validation: 52 samples
- Test: 55 samples
- Split method: source-file grouped and approximately log-strain-rate stratified
- Clean split checks: PASS
  - no `source_file` overlap across train / validation / test
  - no `source_global_index` overlap across train / validation / test
  - no zero-offset or negative-offset rows
  - no contradictory input-target pairs

## Test Metrics

Mean +/- standard deviation over 3 random seeds.

| Model | Training samples | Parameters | Test RMSE | Test relative L2 | Circular RMSE (deg) | Circular MAE (deg) | RMSE improvement vs persistence |
|---|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 0 | 0 | 0.29755 +/- 0 | 0.464883 +/- 0 | 32.8587 +/- 0 | 9.80081 +/- 0 | 0 +/- 0 |
| ResNet CNN | 411 | 85,313 | 0.176992 +/- 0.0010153 | 0.276527 +/- 0.00158628 | 27.3299 +/- 0.198451 | 11.9517 +/- 0.194865 | 40.5168 +/- 0.341221 |
| FNO | 411 | 593,345 | 0.174774 +/- 0.00434005 | 0.273062 +/- 0.00678076 | 26.8936 +/- 1.09549 | 12.1977 +/- 0.843617 | 41.2622 +/- 1.4586 |

## Interpretation

On this cleaned Euler-1 reduced-data benchmark, FNO has the lowest mean test RMSE, relative L2, and circular RMSE. The ResNet CNN is close and has a slightly lower circular MAE. Both trainable surrogates improve substantially over persistence.
