# Example training data for FNO

In this folder, we provide two example datasets used for FNO training. The naming of the datasets follow the convention:

        [variable_to_train]_S[shear_strain_rate]_H[depth_equivelant_pressure]_T[temperature]_data_train.npy.npz

variable_to_train can be "grain_kde", "euler_1", "euler_2", and "euler_3".

The two example datasets are the temporal evolution of a 2D ice microstructure over 4000 days with a time step of 20 days. The starting microstructure (ice grain shapes and c-axis distribution) can be found in the Supplement in Liu et al., 2026 (under review).

For example, `euler_1_S1800000e-14_H1000_T-1.0_data_train.npy.npz` records how the euler 1 angle evolves under shear strain rate is $1.8\times10^{-8}$ 1/s, pressure is $8.8\times10^6$ Pa (equivelant of 1000 m thick ice), and temperature is $-1^oC$.

**Important**: These two example datasets are demonstrative and do not intent to be thorough. They are essentially two Elle simulations with only two sets of controlling parameters (strain rate, pressure, temperature). To train a proper FNO model, you need the training data to represent the thermal and mechanical conditions of the area of interest. This means that you will need to do (a lot) more Elle simulations with possibly different controlling parameters to enrich the datasets.

The Elle image and a short tutorial can be found under Elle folder.

