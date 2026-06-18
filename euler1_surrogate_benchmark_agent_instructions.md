# AI Coding Agent Instructions: Euler-1 Reduced-Data Surrogate Benchmark

## 1. Objective

Build a clean, reproducible Euler-1 surrogate comparison pipeline to address a reviewer concern that the Fourier Neural Operator (FNO) surrogate was not compared against another surrogate architecture.

The benchmark must compare:

1. **Persistence baseline**
2. **Local CNN/ResNet surrogate**
3. **FNO surrogate**

The output should be a reviewer-facing result table and supporting diagnostics, not just trained model checkpoints.

This task is limited to **Euler angle 1**. Label all outputs as:

> Euler-1 reduced-data benchmark

Do **not** present the result as a full benchmark of the complete microscale surrogate, because the full paper surrogate includes grain-area distribution plus three Euler-angle fields.

---

## 2. Scientific Context

The manuscript uses an FNO as a surrogate for the Elle microscale model. The surrogate maps normalized thermomechanical control variables to microstructure evolution:

- pressure;
- temperature;
- strain rate;
- grain-area distribution;
- Euler-angle / fabric fields.

The full manuscript trains four FNO models:

1. one model for grain-area distribution `f(A)`;
2. one model for Euler angle `phi_1`;
3. one model for Euler angle `phi_2`;
4. one model for Euler angle `phi_3`.

This benchmark only uses the Euler-1 data. Therefore, all language in scripts, plots, tables, and written summaries must clearly state that this is a **reduced Euler-1 comparison**, not a full replacement for the original grain-size plus three-Euler-angle surrogate evaluation.

---

## 3. Input Files

Use these HDF5 files as the raw data source:

```text
/mnt/data/euler1_train_data.h5
/mnt/data/euler1_valid_data(1).h5
/mnt/data/euler1_test_data(1).h5
```

Also inspect the current training script:

```text
/mnt/data/train_euler1_ddp(1).py
```

Do **not** regenerate Elle simulations. Do **not** require the original `.npz` files. Work only from the provided HDF5 files unless explicitly instructed otherwise.

---

## 4. Raw HDF5 Data Structure

Each HDF5 file contains normalized fields:

| Dataset | Meaning | Expected shape |
|---|---|---|
| `euler1_known` | input Euler-1 field | `(N, 128, 128, 1)` |
| `euler1_predict` | target Euler-1 field | `(N, 128, 128, 1)` |
| `strain_rate` | normalized strain-rate field | `(N, 128, 128, 1)` |
| `temperature` | normalized temperature field | `(N, 128, 128, 1)` |
| `pressure` | normalized pressure field | `(N, 128, 128, 1)` |

Useful metadata fields include:

| Metadata | Meaning |
|---|---|
| `source_file` | source Elle simulation file |
| `source_S_value` | source strain-rate value |
| `source_global_index` | global sample identifier |
| `known_step_1based` | known/input time step |
| `predict_step_1based` | target/prediction time step |
| `source_step_sequence_1based` | original source time-step sequence |

The raw files are readable and finite, but the current split is **not acceptable for final reporting** without cleaning and regrouping.

---

## 5. Critical Data Problems to Fix

### 5.1 Raw split is too small

The current raw split is:

| Split | Samples |
|---|---:|
| Train | 474 |
| Validation | 42 |
| Test | 11 |

The test set has only 11 samples. This is sufficient for debugging, but too small for a robust reviewer-facing comparison.

### 5.2 Raw split leaks by source simulation

The raw split is sample-level rather than source-simulation-level. The same `source_file` appears across train, validation, and test. This creates leakage because the model can see neighboring windows from the same underlying Elle simulation during training.

Required fix:

> Concatenate the raw HDF5 files, clean bad rows, and resplit by `source_file`, not by individual sample.

### 5.3 Zero-offset rows exist

Some rows satisfy:

```python
predict_step_1based - known_step_1based == 0
```

These rows must be removed.

### 5.4 Contradictory labels exist

Some samples share the same input identity:

```python
(source_file, known_step_1based)
```

but have multiple different targets. These are contradictory supervised-learning pairs and must not be used.

After removing zero-offset rows, the pipeline must check again for contradictory pairs. If any remain, fail loudly.

### 5.5 Test set is biased

The raw test set is biased toward relatively warm, high-pressure, moderate-to-high-strain-rate cases. This is another reason to construct a source-file-grouped, approximately strain-rate-stratified clean split.

---

## 6. Required Project Structure

Create or update the project structure as follows:

```text
scripts/
  inspect_euler1_h5.py
  prepare_clean_euler1_splits.py
  compare_euler1_surrogates.py

src/
  euler1_data.py
  euler1_models.py
  euler1_metrics.py

results/
  euler1_comparison/

data/
  euler1_clean/
```

---

# Deliverable 1: Data Inspection Script

## Script

```text
scripts/inspect_euler1_h5.py
```

## Required CLI

```bash
python scripts/inspect_euler1_h5.py \
  --h5 /mnt/data/euler1_train_data.h5 \
       '/mnt/data/euler1_valid_data(1).h5' \
       '/mnt/data/euler1_test_data(1).h5' \
  --out results/euler1_comparison/data_report_raw.json
```

## Required checks

For each split, report:

- sample count;
- dataset names;
- shape and dtype for each dataset;
- min, max, mean, and standard deviation for numeric arrays;
- finite / NaN / Inf checks;
- unique `source_file` count;
- unique `source_global_index` count;
- known-step and predict-step ranges;
- offset distribution:

```python
predict_step_1based - known_step_1based
```

- number of zero-offset or negative-offset rows;
- number of duplicate exact rows:

```python
(source_file, known_step_1based, predict_step_1based)
```

- number of contradictory input rows:

```python
(source_file, known_step_1based)
```

with multiple targets;

- `source_file` overlap among train / validation / test;
- `source_global_index` overlap among train / validation / test.

## Required summary behavior

The script must print a clear `PASS` / `WARN` / `FAIL` summary.

Expected result on the raw uploaded data:

| Check | Expected status |
|---|---|
| Required arrays exist | PASS |
| Required arrays are finite | PASS |
| Source-file leakage across splits | WARN or FAIL |
| Zero-offset rows exist | FAIL |
| Contradictory labels exist | FAIL |
| Test set has only 11 samples | WARN |

---

# Deliverable 2: Clean Split Construction

## Script

```text
scripts/prepare_clean_euler1_splits.py
```

## Required CLI

```bash
python scripts/prepare_clean_euler1_splits.py \
  --input_h5 /mnt/data/euler1_train_data.h5 \
             '/mnt/data/euler1_valid_data(1).h5' \
             '/mnt/data/euler1_test_data(1).h5' \
  --out_dir data/euler1_clean \
  --seed 12345 \
  --train_ratio 0.8 \
  --valid_ratio 0.1 \
  --test_ratio 0.1 \
  --stratify_by logS
```

## Required behavior

1. Load and concatenate all rows from the three raw HDF5 files.
2. Preserve all datasets and metadata where possible.
3. Remove rows with:

```python
predict_step_1based <= known_step_1based
```

4. Check for repeated input states:

```python
(source_file, known_step_1based)
```

with more than one target. If any remain after removing zero-offset / negative-offset rows, fail loudly.

5. Group rows by `source_file`.
6. Split groups, not individual rows, into train / validation / test.
7. Stratify approximately by:

```python
log10(source_S_value)
```

using 4 or 5 bins.

8. Write:

```text
data/euler1_clean/euler1_train_clean.h5
data/euler1_clean/euler1_valid_clean.h5
data/euler1_clean/euler1_test_clean.h5
```

9. Add HDF5 attributes:

```text
split_name
split_method = "source_file_grouped_logS_stratified"
split_seed
removed_zero_or_negative_offset_count
source_file_count
sample_count
```

10. After writing, run the same inspection checks and assert:

- no `source_file` overlap across splits;
- no `source_global_index` overlap across splits;
- no zero-offset or negative-offset rows;
- no contradictory `(source_file, known_step_1based)` rows;
- all required arrays are finite.

Do **not** use the original raw split for final reported results.

---

# Deliverable 3: Dataset Loader

## Module

```text
src/euler1_data.py
```

## Required class

```python
Euler1H5Dataset(h5_path, preload=False, include_delta_step=True)
```

Each item must return a dictionary:

```python
{
    "x": euler1_known,
    "y": euler1_predict,
    "strain_rate": strain_rate,
    "temperature": temperature,
    "pressure": pressure,
    "known_step": known_step_1based,
    "predict_step": predict_step_1based,
    "delta_step": predict_step_1based - known_step_1based,
    "source_file": source_file,
    "source_global_index": source_global_index,
}
```

## Model input tensor

The model input feature tensor must include:

1. Euler-1 known field;
2. strain-rate field;
3. temperature field;
4. pressure field;
5. x-coordinate grid;
6. y-coordinate grid;
7. normalized `delta_step` as a constant channel.

The delta-step channel is required because this dataset uses variable prediction offsets.

Normalize `delta_step` consistently. For example:

```python
delta_step_norm = 2 * (delta_step - delta_min) / (delta_max - delta_min) - 1
```

Compute `delta_min` and `delta_max` from the training split only. Reuse those values for validation and test.

---

# Deliverable 4: Models

## Module

```text
src/euler1_models.py
```

## Required model 1: `PersistenceBaseline`

Behavior:

- returns the known Euler-1 field unchanged;
- has no trainable parameters;
- is evaluated on the same splits as the neural surrogates.

## Required model 2: `ResNetCNNSurrogate`

This is the main local convolutional baseline.

Requirements:

- input channels:
  - `euler1_known`;
  - `strain_rate`;
  - `temperature`;
  - `pressure`;
  - `x_grid`;
  - `y_grid`;
  - `delta_step_norm`;
- 6 to 8 convolutional layers, or 4 residual blocks;
- width 32 or 64;
- kernel size 3;
- one output channel;
- `tanh` output activation so predictions stay in `[-1, 1]`.

Do not rely only on a tiny CNN with about 3,000 parameters for the reviewer-facing comparison. Include a moderate ResNet CNN so the comparison is credible.

## Required model 3: `FNO2dSurrogate`

Requirements:

- use the existing FNO idea but make the input-channel count explicit;
- use 4 active spectral blocks;
- do not declare 7 blocks if only 4 are used;
- remove unused layers or make parameter counting count only active layers;
- do not apply `torch.nn.functional.normalize` to already normalized control channels inside the model;
- use `tanh` output activation;
- make Fourier modes and width configurable:

```text
--fno_modes 12 or 24
--fno_width 16
```

## Optional models

Optional additional baselines may include:

- `PointwiseMLP`;
- `LinearPointwise`.

These are useful as sanity checks but should not replace the ResNet CNN comparison.

---

# Deliverable 5: Metrics

## Module

```text
src/euler1_metrics.py
```

Report all metrics on train, validation, and test:

- MSE;
- RMSE in normalized units;
- MAE;
- relative L2;
- circular RMSE in degrees;
- circular MAE in degrees;
- parameter count;
- training time;
- inference time per sample.

## Circular metrics

Euler-1 is normalized to `[-1, 1]`, corresponding to `[-180°, 180°]`.

Use:

```python
diff_rad = atan2(sin(pi * (pred - target)), cos(pi * (pred - target)))
circular_rmse_deg = sqrt(mean(diff_rad ** 2)) * 180 / pi
```

Also report improvement over persistence:

```python
rmse_improvement_percent = 100 * (persistence_rmse - model_rmse) / persistence_rmse
```

---

# Deliverable 6: Training and Comparison Script

## Script

```text
scripts/compare_euler1_surrogates.py
```

## Required CLI

```bash
python scripts/compare_euler1_surrogates.py \
  --train_h5 data/euler1_clean/euler1_train_clean.h5 \
  --valid_h5 data/euler1_clean/euler1_valid_clean.h5 \
  --test_h5 data/euler1_clean/euler1_test_clean.h5 \
  --models persistence,resnet_cnn,fno \
  --epochs 100 \
  --batch_size 8 \
  --learning_rate 1e-3 \
  --seeds 12345 23456 34567 \
  --output_dir results/euler1_comparison
```

## Training requirements

- Train every trainable model on the same cleaned train split.
- Use the same validation split for early stopping.
- Use the same test split for final metrics.
- Save the best checkpoint by validation RMSE or validation circular RMSE.
- Use early stopping with patience 20.
- Use gradient clipping with max norm 10.
- Use MSE loss by default.
- Optional: add a smoothness penalty, but if used, apply it to both FNO and CNN so the comparison is fair.

## Required output files

```text
results/euler1_comparison/summary_by_seed.csv
results/euler1_comparison/summary_mean_std.csv
results/euler1_comparison/history.csv
results/euler1_comparison/data_report_raw.json
results/euler1_comparison/data_report_clean.json
results/euler1_comparison/test_predictions_sampled.npz
results/euler1_comparison/best_fno.pt
results/euler1_comparison/best_resnet_cnn.pt
```

## Required plots

Create the following plots:

- train / validation loss curves;
- test RMSE bar plot with mean ± standard deviation over seeds;
- circular RMSE bar plot with mean ± standard deviation over seeds;
- representative target / prediction / error maps for persistence, CNN, and FNO;
- scatter plot of predicted vs. true normalized Euler-1 values for the test set.

---

# Deliverable 7: Acceptance Criteria

The final result is acceptable only if all of the following are true:

1. The clean split has no `source_file` overlap among train / validation / test.
2. The clean split has no `source_global_index` overlap among train / validation / test.
3. The clean split has no zero-offset or negative-offset samples.
4. The clean split has no contradictory input-target pairs.
5. The scripts can run from a clean checkout using explicit HDF5 paths.
6. The test metrics are computed only once per trained seed, after model selection by validation.
7. The result table includes persistence, ResNet CNN, and FNO.
8. The result table reports mean ± standard deviation over at least 3 random seeds.
9. The final written summary clearly states that this is an Euler-1 reduced-data benchmark, not a full replacement for the manuscript’s grain-size plus three-Euler-angle surrogate evaluation.

---

# Recommended Workflow

## Step 1: Inspect raw files

```bash
python scripts/inspect_euler1_h5.py \
  --h5 /mnt/data/euler1_train_data.h5 \
       '/mnt/data/euler1_valid_data(1).h5' \
       '/mnt/data/euler1_test_data(1).h5' \
  --out results/euler1_comparison/data_report_raw.json
```

Confirm that the raw report identifies:

- valid numerical arrays;
- source-file leakage;
- zero-offset rows;
- contradictory labels;
- small raw test set.

## Step 2: Prepare clean splits

```bash
python scripts/prepare_clean_euler1_splits.py \
  --input_h5 /mnt/data/euler1_train_data.h5 \
             '/mnt/data/euler1_valid_data(1).h5' \
             '/mnt/data/euler1_test_data(1).h5' \
  --out_dir data/euler1_clean \
  --seed 12345 \
  --train_ratio 0.8 \
  --valid_ratio 0.1 \
  --test_ratio 0.1 \
  --stratify_by logS
```

Confirm that the clean report passes all leakage and offset checks.

## Step 3: Run comparison

```bash
python scripts/compare_euler1_surrogates.py \
  --train_h5 data/euler1_clean/euler1_train_clean.h5 \
  --valid_h5 data/euler1_clean/euler1_valid_clean.h5 \
  --test_h5 data/euler1_clean/euler1_test_clean.h5 \
  --models persistence,resnet_cnn,fno \
  --epochs 100 \
  --batch_size 8 \
  --learning_rate 1e-3 \
  --seeds 12345 23456 34567 \
  --output_dir results/euler1_comparison
```

## Step 4: Review results

Check that:

- FNO, CNN, and persistence are evaluated on the exact same cleaned test split;
- the comparison includes both ordinary RMSE and circular angular metrics;
- FNO is compared against a credible CNN/ResNet baseline, not only a tiny CNN;
- plots are representative and not cherry-picked;
- parameter counts reflect active trainable parameters only.

---

# Implementation Notes

## HDF5 handling

Use `h5py`. Avoid loading full arrays repeatedly inside a training loop unless `preload=True` is explicitly selected. For small benchmark runs, preloading may be acceptable, but the implementation should support non-preloaded reads.

## Tensor layout

The HDF5 arrays are stored as:

```text
(N, 128, 128, 1)
```

PyTorch convolution models usually expect:

```text
(N, C, H, W)
```

Implement a consistent conversion in the dataset or collate function.

## Coordinates

Create coordinate grids with values in `[-1, 1]`:

```python
x = torch.linspace(-1, 1, W)
y = torch.linspace(-1, 1, H)
```

Broadcast them to channels of shape `(1, H, W)` and concatenate to the input tensor.

## Delta-step normalization

Because prediction offsets vary, include `delta_step_norm` as a constant channel. Compute min/max from the cleaned training split and store them in the experiment metadata.

## Loss function

Use MSE by default for training. Evaluate with both ordinary and circular metrics.

If a smoothness penalty is added, apply the same penalty to both FNO and CNN:

```python
loss = mse_loss + lambda_smooth * smoothness_loss
```

Do not apply a smoothness penalty only to FNO or only to CNN.

## Model selection

Use validation metrics only for early stopping and checkpoint selection. Do not tune based on test performance.

---

# Suggested Final Result Table

The reviewer-facing table should look like this:

| Model | Training samples | Parameters | Test RMSE | Test relative L2 | Circular RMSE | Circular MAE | RMSE improvement vs persistence |
|---|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 0 trainable | 0 | mean ± std | mean ± std | mean ± std | mean ± std | 0% |
| ResNet CNN | cleaned train split | active trainable count | mean ± std | mean ± std | mean ± std | mean ± std | value |
| FNO | cleaned train split | active trainable count | mean ± std | mean ± std | mean ± std | mean ± std | value |

Use `mean ± standard deviation` over at least 3 random seeds for trainable models.

---

# Suggested Rebuttal / Manuscript Language

Use wording like:

> To address the reviewer’s concern, we performed a reduced-data benchmark for the Euler-1 microscale surrogate. We compared the FNO against a local convolutional ResNet baseline and a persistence baseline using the same cleaned, source-file-disjoint Elle-derived HDF5 split. The comparison evaluates whether the FNO advantage persists when all models are trained and tested on the same reduced dataset.

Avoid wording like:

> We selected a worse model.

Also avoid claiming that this Euler-1-only experiment benchmarks the entire microscale surrogate. The correct framing is:

> This is a reduced Euler-1 benchmark that complements the full FNO-vs-Elle validation already reported for grain size and Euler-angle fields.

---

# Do Not Do

Do **not**:

- report final results from the raw leaky split;
- train on zero-offset rows;
- train on contradictory input-target pairs;
- compare FNO trained on one split against CNN trained on a different split;
- tune hyperparameters based on test metrics;
- describe the baseline as intentionally worse;
- rely only on a tiny CNN for the final reviewer-facing comparison;
- claim this Euler-1-only benchmark evaluates the full grain-size plus fabric surrogate.

---

# Bottom Line

The uploaded HDF5 files are valuable raw material, but the current split must be cleaned and regrouped by source simulation before use. The biggest required fixes are:

1. remove zero-offset and contradictory rows;
2. resplit by `source_file`;
3. add explicit test evaluation;
4. include persistence, ResNet CNN, and FNO;
5. report mean ± standard deviation over multiple seeds;
6. clearly label the result as an Euler-1 reduced-data benchmark.
