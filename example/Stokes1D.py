import numpy as np
import time
import argparse

import sys
sys.path.append('../src/')

import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
from model_euler import FNO2d
from model_grain_kde import FNO1d
from utilities3_grain import log_normalize, reference_normalize
from utilities3_euler import euler_to_orientation_tensor

# Constants
S2Y = 3600 * 24 * 365.25
S2D = 3600 * 24
PI = np.pi

################## Function Definitions ##################
class GoldsbyFlowLaw:
    def __init__(self, device: str = "cuda"):
        self.device = device
        # Common parameters
        self.R = 8.314  # Gas constant
        self.litmax = 200
        self.tol = 1e-13
        self.rele = 5e-1

        # ---------- Diffusion creep (Eq. 4 + Table 6) ----------
        # Eq (4) uses diffusion coefficients D = D0 * exp(-Q/RT)
        self.n_diff = 1.0
        # Table 6 parameters (SI)
        self.Vm = 1.97e-5          # m^3/mol (Table 6)
        self.D0_v = 9.10e-4        # m^2/s (preexp volume diffusion)
        self.Qv = 59.4e3           # J/mol (activation energy volume diffusion)
        self.delta = 9.04e-10      # m (grain boundary width)
        # Boundary diffusion: Qb given; D0_b is estimated/assumed ~ D0_v in their discussion
        self.Qb = 49e3             # J/mol (activation energy boundary diffusion)
        self.D0_b = 8.4e-4         # m^2/s (their estimated upper bound ~8.4e-4, sec. 5.5 in their paper) 

        # ---------- Basal slip–accommodated GBS (Eq. 3 + Table 5) ----------
        self.n_basal = 2.4
        self.p_basal = 0.0
        # A_basal given in MPa^-n s^-1 in Table 5; convert to Pa^-n s^-1:
        # A_Pa = A_MPa * (1e6)^(-n) = A_MPa * 10^(-6n)
        self.A_basal = 5.5e7 * 10 ** (-6 * self.n_basal)
        self.Q_basal = 60e3         # J/mol

        # ---------- GBS-accommodated basal slip (n=1.8, p=1.4) ----------
        self.n_gbs = 1.8
        self.p_gbs = 1.4
        # Two thermal regimes (low/high); Table 5 uses MPa^-n m^p s^-1 -> convert stress unit only
        self.A_gbs_hi = 3e26 * 10 ** (-6 * self.n_gbs)
        self.Q_gbs_hi = 192e3
        self.A_gbs_lo = 3.9e-3 * 10 ** (-6 * self.n_gbs)
        self.Q_gbs_lo = 49e3
        self.Tstar_gbs = 255.0      # K 
        
        # ---------- Dislocation creep (n=4.0) ----------
        self.n_disl = 4.0
        self.p_disl = 0.0
        self.A_disl_hi = 6e28 * 10 ** (-6 * self.n_disl)
        self.Q_disl_hi = 180e3
        self.A_disl_lo = 4.0e5 * 10 ** (-6 * self.n_disl)
        self.Q_disl_lo = 60e3
        self.Tstar_disl = 258.0     # K (premelting enhancement threshold)

        # Pre-compute F factors
        self.F_diff     = (2**((2*self.n_diff-1)/self.n_diff))**(0.9) # simple shear epxeriments (not sure 100%)
        self.F_basal    = (2**((self.n_basal-1)/self.n_basal)*3**((self.n_basal+1)/(2*self.n_basal)))**(0.9) # pure shear epxeriments
        self.F_gbs  = (2**((self.n_gbs - 1) / self.n_gbs) * 
                      3**((self.n_gbs + 1) / (2 * self.n_gbs)))**(0.9)
        self.F_disl = (2**((self.n_disl - 1) / self.n_disl) * 
                       3**((self.n_disl + 1) / (2 * self.n_disl)))**(0.9)
    @staticmethod
    def _to_numpy(x):
        # robustly convert torch->numpy if needed, without importing torch
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def compute_viscosity(
        self,
        Mus,         # viscosity from last time (same shape)
        EII2,        # second invariant of strain rate tensor squared (s^-2)
        TII2,
        T,           # temperature (K)
        P,           # pressure (Pa)  (sign convention matters; see ph usage)
        d,           # grain size (m)
        w13,         # weakening factor
        time_step,
        ph=7e-8,     # K/Pa pressure-melting coefficient magnitude
        pressure_lowers_melting=True,
        eps_floor=1e-30
    ):
        """
        Returns:
            etan : effective viscosity (Pa s)
            mech : diagnostic dominant mechanism label (same as your scheme)
        """

        Mus = self._to_numpy(Mus)
        EII2 = self._to_numpy(EII2)
        T = self._to_numpy(T)
        P = self._to_numpy(P)
        d = self._to_numpy(d)
        w13 = self._to_numpy(w13)
        
        # --- homologous temperature correction ---
        # Pressure *lowers* melting point if P is positive compressive; adjust sign accordingly.
        if pressure_lowers_melting:
            T_h = T - ph * P
        else:
            T_h = T + ph * P
        # avoid division by zero
        T_h = np.maximum(T_h, 1.0)
        d = np.maximum(d, 1e-12)
        
        # strain-rate invariant
        EII = np.sqrt(np.maximum(EII2, eps_floor))
        TII = np.sqrt(TII2)
        # ---- Diffusion coefficients: D = D0 exp(-Q/RT) ---- 
        Dv = self.D0_v * np.exp(-self.Qv / (self.R * T_h))
        Db = self.D0_b * np.exp(-self.Qb / (self.R * T_h))
        # Diffusion creep coefficient C_diff in eps = C_diff * tau (n=1)
        # Eq. (4) gives diffusion creep rate proportional to stress and includes a (Dv + boundary term)
        # We implement: eps_diff = (42 * Vm / (R*T)) * (Dv/d^2 + (np.pi * delta * Db)/d^3) * tau
        C_diff = (42.0 * self.Vm / (self.R * T_h)) * (Dv / d**2 + (np.pi * self.delta * Db) / d**3) * self.F_diff

        # ---- Basal term (Eq. 1 form) ---- 
        C_basal = self.A_basal * np.exp(-self.Q_basal / (self.R * T_h)) * self.F_basal  # p=0

        # ---- GBS term: pick regime by its own T*  ----
        gbs_hi = T_h >= self.Tstar_gbs
        A_gbs = np.where(gbs_hi, self.A_gbs_hi, self.A_gbs_lo)
        Q_gbs = np.where(gbs_hi, self.Q_gbs_hi, self.Q_gbs_lo)
        C_gbs = A_gbs * d**(-self.p_gbs) * np.exp(-Q_gbs / (self.R * T_h)) * self.F_gbs

        # ---- Dislocation term: two regimes; high-T enhancement via premelting ---- 
        disl_hi = T_h >= self.Tstar_disl
        A_disl = np.where(disl_hi, self.A_disl_hi, self.A_disl_lo)
        Q_disl = np.where(disl_hi, self.Q_disl_hi, self.Q_disl_lo)
        C_disl = A_disl * np.exp(-Q_disl / (self.R * T_h)) * self.F_disl

        # ---- Initial guess for eta from single-mechanism inversions (stable) ----
        # For eps = C * tau^n, tau = (eps/C)^(1/n), eta = tau/(2 eps) = 0.5 * C^(-1/n) * eps^(1/n - 1).
        def eta_single(C, n):
            Csafe = np.maximum(C, eps_floor)
            return 0.5 * (Csafe ** (-1.0 / n)) * (EII ** (1.0 / n - 1.0))
        
        eta_diff = eta_single(C_diff, self.n_diff)
        eta_basal = eta_single(C_basal, self.n_basal)
        eta_gbs = eta_single(C_gbs, self.n_gbs)
        eta_disl = eta_single(C_disl, self.n_disl)
        # eta_0 = (eta_diff + eta_disl + (1/eta_basal + 1/eta_gbs)**(-1))
        # bracket-like guess 
        Tii_basal = 2.0 * eta_basal * EII
        Tii_gbs = 2.0 * eta_gbs * EII
        eta_tmp = (Tii_basal + Tii_gbs) / (2.0 * EII)

        eta_up = np.minimum(np.minimum(eta_diff, eta_tmp), eta_disl)
        eta_lo = ((eta_basal + eta_gbs) /
                  (eta_basal / np.maximum(eta_disl, eps_floor) +
                   eta_basal / np.maximum(eta_diff, eps_floor) + 1.0 +
                   eta_gbs / np.maximum(eta_disl, eps_floor) +
                   eta_gbs / np.maximum(eta_diff, eps_floor)))
        # check if high and low cross zeros, thus is valid space to find solution
        tau_lo = 2.0 * eta_lo * EII
        tau_up = 2.0 * eta_up * EII
        
        eps_diff_lo = C_diff * tau_lo**self.n_diff
        eps_basal_lo = C_basal * tau_lo**self.n_basal
        eps_gbs_lo = C_gbs * tau_lo**self.n_gbs
        eps_disl_lo = C_disl * tau_lo**self.n_disl
        eps_diff_up = C_diff * tau_up**self.n_diff
        eps_basal_up = C_basal * tau_up**self.n_basal
        eps_gbs_up = C_gbs * tau_up**self.n_gbs
        eps_disl_up = C_disl * tau_up**self.n_disl

        eps_total_lo = eps_diff_lo + (eps_basal_lo**(-1) + eps_gbs_lo**(-1))**(-1) + eps_disl_lo
        eps_total_up = eps_diff_up + (eps_basal_up**(-1) + eps_gbs_up**(-1))**(-1) + eps_disl_up

        # print(f'Guess lo: {[f"{x:.4e}" for x in eps_total_lo[:6]]}')
        # print(f'Guess up: {[f"{x:.4e}" for x in eps_total_up[:6]]}')
        # print(f'True: {[f"{x:.4e}" for x in EII[:6]]}')
        
        eta_0 = 0.5 * (eta_up + eta_lo)
        # print(f'eta_up: {eta_up.min()}, eta_lo: {eta_lo.min()}, eta_0: {eta_0.min()}')
        
        # ---- Newton solve for eta so that EII = eps_total(tau) ----
        res0 = None
        for lit in range(self.litmax):
            # starting guess
            tau = 2.0 * eta_0 * EII  # tau_II

            # mechanism strain rates
            eps_diff = C_diff * tau**self.n_diff
            eps_basal = C_basal * tau**self.n_basal
            eps_gbs = C_gbs * tau**self.n_gbs
            eps_disl = C_disl * tau**self.n_disl
            # enforce floors
            eps_basal = np.maximum(eps_basal, eps_floor)
            eps_gbs   = np.maximum(eps_gbs,   eps_floor)

            # total strain rate giving the current guess of viscosity
            eps_total = eps_diff + (eps_basal**(-1) + eps_gbs**(-1))**(-1) + eps_disl
            r = EII - eps_total
            res = np.linalg.norm(r.ravel(), ord=2)
            if res0 is None:
                res0 = max(res, eps_floor)
            if res / res0 < self.tol:
                break

            # derivative dr/deta via tau = 2 eta EII
            dtau_deta = 2.0 * EII
            # d(eps_i)/deta = d(eps_i)/dtau * dtau/deta = (n C tau^(n-1)) * (2 EII)
            deps_diff = (self.n_diff * C_diff * tau**(self.n_diff - 1.0)) * dtau_deta
            deps_disl = (self.n_disl * C_disl * tau**(self.n_disl - 1.0)) * dtau_deta
            # two harmonic terms' derivative
            deps_gbs_basal = -(tau**(-self.n_basal)/C_basal  + tau**(-self.n_gbs)/C_gbs)**(-2) \
                            * (-self.n_basal/C_basal*tau**(-self.n_basal-1) - self.n_gbs/C_gbs*tau**(-self.n_gbs-1)) * dtau_deta
            
            dr_deta = -(deps_diff + deps_gbs_basal + deps_disl)
            
            # newton updates
            # eta_0 = eta_0 - r/dr_d
            dr_deta_safe = np.where(np.abs(dr_deta) > eps_floor, dr_deta, np.sign(dr_deta) * eps_floor)
            eta_0 = np.maximum(eta_0 - r / dr_deta_safe, eps_floor)
        
        # print(f'final goldsby viscosity range: {eta_0.min():.4e},{eta_0.max():.4e}, eps res: {res:.4e}')
        
        # blend with previous viscosity 
        etan = w13 * np.exp(self.rele * np.log(np.maximum(eta_0, eps_floor)) +
                            (1.0 - self.rele) * np.log(np.maximum(Mus, eps_floor)))
        return etan

def saveResults2NPZ(Vxe, P, T, epsxz, etav, d, grain_kde, zv, 
                        euler1_all,euler2_all,euler3_all,
                        eigenvalues, w13,filepath, filename):
    # euler1_all = euler1_all.detach().numpy()
    # euler2_all = euler2_all.detach().numpy()
    # euler3_all = euler3_all.detach().numpy()
    data = {
        "Vxe": Vxe,
        "Pressure": P,
        "Temperature": T,
        "Strain_rate": epsxz,
        "viscosity": etav,
        "Grain_size": d,
        "Depth": zv,
        "Grain_kde":grain_kde,
        "eigenvalues":eigenvalues,
        "weaken_factor":w13,
        "Euler1_2d":euler1_all,
        "Euler2_2d":euler2_all,
        "Euler3_2d":euler3_all
    }
    np.savez(f"{filepath}{filename}", **data)

########################################################################


@torch.no_grad()
def ice_column1D(grain_model_epoch,grain_model_path,euler_model_epoch,euler_model_path,save_data_path):
    # Constants
    S2Y = 3600 * 24 * 365.25
    S2D = 3600 * 24
    PI = np.pi
    
    # Parameters
    MULTISCALE = False
    ADVECT = True
    H = 1000.0
    alpha = 0.8
    alpha_rad = alpha * PI / 180
    rho = 900
    g = -9.80665
    eta0 = 1e15
    T_surf = 247.15 # -26
    T_bed = 272.15 # 273.15 - 0
    T_K = 273.15
    kappa = 2.51
    cp = 2096.9
    dt = 20 * S2D
    nt = 500
    c = 3.0
    nz = 63
    tol = 1e-10
    iterMax = 1e5+1
    nout = iterMax - 1
    dz = H / nz
    damp = 0.4 #1.0 - 0.1 / nz
    dtaudT  = 1 / (1/(dz**2 / kappa * rho * cp)/4.1 + 1/dt)
        
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # device should be cuda
    print(f'device: {device}')
    
    # ----------- FNO kde set up ----------#
    # IMPORTANT: check .out to see what reference max and min were used in training
    kde_max, kde_min = 0.0337, 0
    H_max, H_min = 1000*9.80665*900, 1*9.80665*900
    S_max, S_min = 1800000e-14,21e-14
    
    # the model parameter MUST be the same as the training
    step_known_kde  = 2
    kde_reso        = 256
    mode1           = 64
    width           = 12
    activation_func = 'tanh'
    loss_func       = 'L2'
    grain_model = FNO1d(mode1, width,step_known_kde,activation_func,loss_func)
    grain_model.load_state_dict(torch.load(grain_model_path+str(grain_model_epoch)+'.pth'))
    grain_model = grain_model.to(device)
        
    grains_in = np.zeros((nz + 1,kde_reso,step_known_kde))
    # load initial condition for small scale simulations. This is pre-saved using the scripts in grain_euler_process notebook in postprocess
    init_kde =  torch.load('./../data/syn_init_kde.pt').to(device)
    grains_in  = reference_normalize(init_kde[:,:,:step_known_kde], kde_max, kde_min)
    
    # grain size range (this is pre-defined)
    area_per_pixel = 70*70/128/128
    min_grainsize,max_grainsize = 3, 2628
    grainsize_range = torch.linspace(min_grainsize,max_grainsize, kde_reso).to(device)*area_per_pixel
    grainsize_range = grainsize_range.to(device)
    # Reshape grainsize_range to shape (1, 256, 1) for broadcasting
    grainsize_range = grainsize_range.view(1, -1, 1)
    d = np.ones((nz+1))*6 # d in mm
    
    # ----------- FNO 2D set up ----------#
    step_known_euler= 40
    grid_size_euler = 128
    mode1           = 12
    mode2           = 12
    width           = 16
    activation_func = 'tanh'
    loss_func       = 'L2'
    euler1_model = FNO2d(mode1, mode2, width,step_known_euler,activation_func,loss_func)
    euler1_model.load_state_dict(torch.load(euler_model_path+str(euler_model_epoch)+'.pth'))
    euler2_model = FNO2d(mode1, mode2, width,step_known_euler,activation_func,loss_func)
    euler2_model.load_state_dict(torch.load(euler_model_path.replace('euler1', 'euler2')+str(euler_model_epoch)+'.pth'))
    euler3_model = FNO2d(mode1, mode2, width,step_known_euler,activation_func,loss_func)
    euler3_model.load_state_dict(torch.load(euler_model_path.replace('euler1', 'euler3')+str(euler_model_epoch)+'.pth'))
    euler1_model = euler1_model.to(device)
    euler2_model = euler2_model.to(device)
    euler3_model = euler3_model.to(device)
        
    euler1_in = np.zeros((nz + 1,grid_size_euler,grid_size_euler,step_known_euler))
    euler2_in = np.zeros((nz + 1,grid_size_euler,grid_size_euler,step_known_euler))
    euler3_in = np.zeros((nz + 1,grid_size_euler,grid_size_euler,step_known_euler))
    init_euler1 = torch.load('./../data/syn_init_euler_1.pt').to(device)
    init_euler2 = torch.load('./../data/syn_init_euler_2.pt').to(device)
    init_euler3 = torch.load('./../data/syn_init_euler_3.pt').to(device)
    euler1_in = init_euler1[:,:,:,:step_known_euler]/180
    euler2_in = (init_euler2[:,:,:,:step_known_euler]-45)/45
    euler3_in = init_euler3[:,:,:,:step_known_euler]/180
    
    w13 = np.ones(nz+1) # weakening factor
    eigenvalues=np.ones((nz+1,3))
    scale = np.sqrt((1+400+400)/4) 
    # ----------- FNO set up done ----------#
     
    
    # --------- goldsby initialization
    goldsby = GoldsbyFlowLaw(device)
    
    # --------- goldsby initialization
    # Array initialization
    d_o = np.ones((nz+1,step_known_kde))
    Vxe = np.zeros(nz + 2)
    epsxz = np.zeros(nz + 1)
    taoxz = np.zeros(nz + 1)
    Eii2 = np.zeros(nz + 1)
    Tii2 = np.ones(nz + 1)
    dVxdtau = np.zeros(nz)
    dtauVx = np.zeros(nz)
    Fx = np.zeros(nz)
    
    err_values = np.empty((nt+1,int(iterMax)))
    errT_values = np.empty((nt+1,int(iterMax)))
    errd_values = np.empty(nt+1)
    
    # Initial conditions
    zv = np.linspace(-H, 0, nz + 1)
    etav = eta0 * np.ones(nz + 1)
    # choose initial temperature field. Convergence time may differ based on the temp init
    a = T_bed/200
    b = 30/H*np.log(T_surf/T_bed)
    c = np.log(1/a*(T_bed-T_surf)/(np.exp(-b*H)-1))
    T = a*np.exp(b*zv+c)+273.15 - 27.5 
    T0,Tend = T[0],T[-1]
    qzT = kappa * np.diff(np.diff(T) / dz)/dz
    dTdt = (qzT)/rho/cp
    P = rho * g * zv 
            
    # Time loop
    if MULTISCALE:
        filename = f"ice1D_kde+euler_alpha{alpha}_Tbed{T_bed-T_K}_reso{nz}_step0"
    else:
        filename = f"ice1D_ref_alpha{alpha}_Tbed{T_bed-T_K}_reso{nz}_step0"
    
    
    print(f'---------------------------------------------------------------------')
    print(f'---- FNO related ----')
    print(f'grain kde model: {grain_model_path}{grain_model_epoch}.pth')
    print(f'euler angle model: {euler_model_path}{euler_model_epoch}.pth and others')
    print(f"FNO inputs shapes: grains_in: {grains_in.shape}, euler: {euler1_in.shape}")
    print(f'starting weakening factor w13: {w13}')
    print(f"  euler 1 range: [{euler1_in.min()},{euler1_in.max()}]")
    print(f"  euler 2 range: [{euler2_in.min()},{euler2_in.max()}]")
    print(f"  euler 3 range: [{euler3_in.min()},{euler3_in.max()}]")
    print(f"  grain range  : [{grains_in.min()},{grains_in.max()}]")
    print(f'---- flow model related ----')
    print(f'num of time steps: {nt}. slope: {alpha}')
    print(f'ice thickness H: {H}. Num grid points: {nz}. dz: {dz}')
    print(f'temperature dt: {dtaudT}')
    print(f'Num of iter per step: {iterMax}. threshold: {tol}')
    print(f'---------------------------------------------------------------------')

    start_time = time.time()
    for time_step in range(1, nt + 1):
        T_o, dTdt_o =  T.copy(), dTdt.copy()
        iter = 1
        err = 2 * tol
        errd = 2* err
        while (err > tol) and iter < iterMax:
            epsxz = 0.5 * np.diff(Vxe) / dz 
            Eii2 = epsxz**2 
            # compute viscosity using goldsby
            etav = goldsby.compute_viscosity(etav, Eii2, Tii2, T, P, d.flatten()/1e3, w13, time_step)
            taoxz = 2 * etav * epsxz
            Tii2 = taoxz**2
            Fx = np.diff(taoxz) / dz - rho * g * np.sin(alpha_rad)
            dVxdtau = 0.5 * dVxdtau + 0.5 * (Fx + dVxdtau * damp)
            dtauVx = 1*dz**2 / (0.5 * (etav[:-1] + etav[1:]))
            Vxe[1:-1] += dtauVx * dVxdtau
            # temperature equation
            qzT = kappa * np.diff(np.diff(T) / dz)/dz
            dTdt = (qzT + 2*taoxz[1:-1]*epsxz[1:-1])/rho/cp
            dTdtau = -(T[1:-1] - T_o[1:-1]) / dt + 0.5 * dTdt + 0.5 * dTdt_o # -(T[1:-1] - T_o[1:-1]) / dt + 
            T[1:-1] += dtaudT*dTdtau
            T[T>=T_bed] = T_bed
            # BC
            Vxe[0], Vxe[-1] = -Vxe[1], Vxe[-2]
            T[0], T[-1] = T0,Tend
    
            err = np.linalg.norm(Fx) / nz
            err_values[time_step,iter] = err
            errT_values[time_step,iter] = np.linalg.norm(dTdtau)/nz
            if iter % nout == 0:
                iteration_time = time.time() - start_time
                print(f"time_step={time_step}, iter={iter}, err={err:.2e}, errd={errd:.2e},"
                        f"velocity={Vxe.max()*S2Y:.2e} m/yr, strain_rate={epsxz[0]:.2e} 1/s, viscosity={etav[0]:.2e} PaS, tau={taoxz[0]:.5e},{taoxz[1]:.5e} Pa,"
                        f"grain_size_range=[{d.min():.1f}, {d.max():.1f}] mm, time used={iteration_time/60:.3e} min")
                start_time = time.time()
            iter += 1
            # ------ end of one pseudo iteration ------ #
        if (ADVECT):
            Vx_temp = (Vxe[1:]+Vxe[:-1])/2
            T_temp = (T[1:] + T[:-1])/2
            T[1:-1] -= dt * Vx_temp[1:-1]/dz * (T_temp[1:] - T_temp[:-1])
    
         # update FNO
        if (MULTISCALE):
            S_new = torch.tensor(epsxz.reshape(1, -1), device=device).float()
            H_new = torch.tensor(P.reshape(1, -1), device=device).float()
            T_new = torch.tensor(T.reshape(1, -1), device=device).float() - T_K
            S_new = torch.clamp(S_new, min=S_min, max=S_max).squeeze()
            H_new = torch.clamp(H_new, min=H_min, max=H_max).squeeze()
            # T_new = torch.clamp(T_new, min=T_min_clamp, max=T_max_clamp).squeeze()
            S_new = log_normalize(S_new,S_max, S_min).to(device)  # Shape (64,1)
            H_new = reference_normalize(H_new,H_max, H_min).to(device)  # Shape (64,1)
            T_new = 1/T_new.squeeze(); # Shape (64,1))

            # -------- Below is using FNO ------- #
            # set the parameters outside of training space to the max min of the training space
            strain_in = S_new[:, np.newaxis, np.newaxis].expand(nz+1, kde_reso, step_known_kde)
            temper_in = T_new[:, np.newaxis, np.newaxis].expand(nz+1, kde_reso, step_known_kde)
            pressu_in = H_new[:, np.newaxis, np.newaxis].expand(nz+1, kde_reso, step_known_kde)
            grains_pred  = grain_model(grains_in, strain_in, temper_in,pressu_in)
            grains_in = torch.cat((grains_in[..., 1:], grains_pred), dim=-1)
            # de-normalize kde
            # IMPORTANT: check .out to see what reference max and min were used in training
            kde = (grains_pred+1)/2 * (kde_max - kde_min) + kde_min # de-normalization is correct
            kde = kde / kde.sum(axis=1, keepdims=True) # kde shape [nz+1, kde reso, step known]
            d =  torch.sqrt(torch.sum(grainsize_range.cpu() * kde.cpu(), dim=1)/PI)*2 # d shape [nz+1, step known]
                 
            strain2D_in = S_new[:, np.newaxis, np.newaxis, np.newaxis].expand(nz+1, grid_size_euler,grid_size_euler, step_known_euler)
            temper2D_in = T_new[:, np.newaxis, np.newaxis, np.newaxis].expand(nz+1, grid_size_euler,grid_size_euler, step_known_euler)
            pressu2D_in = H_new[:, np.newaxis, np.newaxis, np.newaxis].expand(nz+1, grid_size_euler,grid_size_euler, step_known_euler)
            euler1_pred = euler1_model(euler1_in, strain2D_in, temper2D_in,pressu2D_in) # euler1_pred shape [nz+1,grid,grid,1]
            euler2_pred = euler2_model(euler2_in, strain2D_in, temper2D_in,pressu2D_in)
            euler3_pred = euler3_model(euler3_in, strain2D_in, temper2D_in,pressu2D_in)
            euler1_in = torch.cat((euler1_in[..., 1:], euler1_pred), dim=-1)
            euler2_in = torch.cat((euler2_in[..., 1:], euler2_pred), dim=-1)
            euler3_in = torch.cat((euler3_in[..., 1:], euler3_pred), dim=-1)
            for i in range(nz+1):
                angle1 = euler1_pred[i,:,:,0].flatten()
                angle2 = euler2_pred[i,:,:,0].flatten()
                angle3 = euler3_pred[i,:,:,0].flatten()
                eigenvalues[i,:],tensor = euler_to_orientation_tensor(angle1*180, angle2*45+45, angle3*180)
                w13[i] = tensor[2,2]/scale
            errd = torch.abs(d.mean() - d_o.mean())/d_o.mean()
            errd_values[time_step]=errd
            # -------- end FNO  ------- #

        # ------------- end one physical step -----------------#

        # Example function to save results - implement according to your requirements
        if MULTISCALE:
            filename = f"ice1D_kde+euler_alpha{alpha}_Tbed{T_bed-T_K}_reso{nz}_step{time_step}"
        else:
            filename = f"ice1D_ref_alpha{alpha}_Tbed{T_bed-T_K}_reso{nz}_step{time_step}"
        
        # saveResults2NPZ(Vxe, P, T, epsxz, etav, d, grains_in, zv, #.cpu().numpy()
        #                     euler1_in, #.cpu().numpy()
        #                     euler2_in, #.cpu().numpy()
        #                     euler3_in, #.cpu().numpy() 
        #                 eigenvalues,w13, #.cpu().numpy()
        #                 save_data_path, filename)
        print(f"\nFinished step {time_step}. Results saved to {save_data_path}{filename}\n")
        # Store convergence data for this time step
        # convergence_data.append({
        #     'time_step': time_step,
        #     'err_values': err_values,
        #     'errd_values': errd_values
        # })
        # ------ end of one time step ------ #
        # ---------------------------------- #

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stokes 1D model for synthetic kde')
    parser.add_argument('--grain_model_epoch', type=int, default=1000, help='number of epoch')
    parser.add_argument('--grain_model_path', type=str, default='./../model/jcp_syn_surfaceSpeed18_grain_kde_smooth1_N9_epoch', help='Path to trained grain kde model')
    parser.add_argument('--euler_model_epoch', type=int, default=500, help='number of epoch')
    parser.add_argument('--euler_model_path', type=str, default='./../model/jcp_syn_surfaceSpeed1.5_euler_2_smooth0.1_N2_epoch', help='Path to trained euler model')
    parser.add_argument('--save_data_path', type=str, default='./../results/', help='Path to saved train data')
    args = parser.parse_args()

    ice_column1D(args.grain_model_epoch,args.grain_model_path,args.euler_model_epoch,args.euler_model_path,args.save_data_path)
