# TwoWayIceModel_Release
Codes for two-way coupling ice model supplement to Liu et al., 2026 (under review).

## Environment

Create the project environment with:

```bash
conda env create -f environment.yml
conda activate twoway-ice
```

This repo's Python entry points expect to be launched from the `example/` directory, for example:

```bash
cd example
python train_euler.py --help
python train_grainsize.py --help
python Stokes1D.py --help
```
