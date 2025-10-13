# run_simulation.py File

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import interp1d

from ASM1_Processes import calculate_process_rates
from bsm1_simulation import bsm1_plant_model, map_particulates_by_tss, particulate_cod_idx, soluble_idx
from bsm1_validation import compute_bsm1_kpis, make_bsm1_validation_plot

os.makedirs('results', exist_ok=True)


def get_bsm1_params():
    """
    Returns a dictionary containing all the kinetic and stoichiometric
    parameters for the BSM1 model, corrected for a temperature of 15 deg C.
    """
    
    # Note: The original BSM1 parameters are defined at 20 deg C.
    # We must apply temperature correction for key kinetic parameters.
    # The Arrhenius equation is used: K_T = K_20 * exp(theta * (T - 20))
    # For BSM1, the standard temperature is T = 15 deg C.
    
    # Operating temperature T = 15  
    
    stoich_params = {
        # Stoichiometric Parameters (from BSM1 doc Table 3)
        'Y_A': 0.24,   # g COD/g N
        'Y_H': 0.67,   # g COD/g COD
        'f_P': 0.08,   # dimensionless
        'i_XB': 0.08,  # g N/g COD
        'i_XP': 0.06,  # g N/g COD
    }

    Kin_params = {
        # Kinetic Parameters (Temperature Corrected)
        'mu_H': 4.0,     # Max specific growth rate for heterotrophs per day
        'K_S': 10.0,      # Substrate half-saturation constant for heterotrophs (g COD/m^3)
        'K_O_H': 0.2,     # Oxygen half-saturation constant for heterotrophs (g -COD/m^3)
        'K_NO': 0.5,      # Nitrate half-saturation constant for heterotrophs (g N/m^3)
        'b_H': 0.3,       # Decay rate for heterotrophs per day
        'eta_g': 0.8,     # Correction factor for anoxic growth of heterotrophs
        'eta_h': 0.8,     # Correction factor for anoxic hydrolysis
        'k_h': 3.0,       # Hydrolysis rate constant per day
        'K_X': 0.1,       # Particulate substrate half-saturation constant (g COD/g COD)
        'mu_A': 0.5,     # Max specific growth rate for autotrophs per day
        'K_NH_A': 1.0,    # Ammonia half-saturation constant for autotrophs (g N/m^3)
        'K_O_A': 0.4,     # Oxygen half-saturation constant for autotrophs (g -COD/m^3)
        'b_A': 0.05,       # Decay rate for autotrophs per day
        'k_a': 0.05,       # Ammonification rate constant m3/(g COD⋅d)
    }
    return stoich_params, Kin_params

def create_influent_data_function(file_path):
    """
    Reads a BSM1 influent data file and returns a function that can
    provide the influent data at any time 't' via interpolation.

    Args:
        file_path (str): The path to the influent data file (e.g., 'influent/Inf_dry_2006.txt').

    Returns:
        function: A function that takes time 't' (in days) as input and returns
                  the flow rate (Q_in) and a numpy array of the 13 influent
                  component concentrations at that time.
    """
    # Load the data file using pandas. It's a tab-separated file with no header.
    df = pd.read_csv(file_path, sep='\t', header=None, skiprows=1,
                     names=['time', 'S_I', 'S_S', 'X_I', 'X_S', 'X_H', 'X_A',
                            'X_P', 'S_O', 'S_NO', 'S_NH', 'S_ND', 'X_ND', 'S_ALK', 'Q_in'])
    
    # Assign column names based on the BSM1 protocol for clarity
    # The first column is time, the second is flow rate (Q), the rest are concentrations.
    column_names = ['time', 'S_I', 'S_S', 'X_I', 'X_S', 'X_H', 'X_A', 'X_P',
                    'S_O', 'S_NO', 'S_NH', 'S_ND', 'X_ND', 'S_ALK', 'Q_in']
    df.columns = column_names
    
    # Extract the time points and the data values
    time_points = df['time'].values

    # Extract the flow rate and concentrations
    # Q_in is the last column.
    # The 13 concentrations are the columns between 'time' and 'Q_in'.
    flow_rate_values = df['Q_in'].values
    concentration_values = df.loc[:, 'S_I':'S_ALK'].values

    # 'bounds_error=False' prevents errors if the solver asks for a time outside the data range.
    # Create an interpolation function for flow rate
    flow_interpolator = interp1d(time_points, flow_rate_values, kind='linear',
                                 bounds_error=False, fill_value=(flow_rate_values[0], flow_rate_values[-1])) # type: ignore
    
    # Create an interpolation function for the 13 concentrations
    conc_interpolator = interp1d(time_points, concentration_values, kind='linear', axis=0,
                                 bounds_error=False, fill_value=(concentration_values[0], concentration_values[-1])) # type: ignore

    def influent_data_function(t):
        return flow_interpolator(t), conc_interpolator(t)

    return influent_data_function

def get_clarifier_params():
    """Returns a dictionary of the Takács clarifier parameters."""
    return {
        'A': 1500,          # Clarifier surface area (m^2)
        'N_layers': 10,     # Number of layers
        'h': 0.4,           # Height of each layer (m) -> Total height = 4m
        'Q_RAS': 18446,     # Underflow rate (m^3/day)
        'Q_w': 385,            # wastage
        'feed_layer': 4,    # Feed enters the 5th layer (0-indexed, 0 = TOP)
        'X_t': 3000.0          # clarification threshold
    }

def get_settling_params():
    """Returns a dictionary of the Takács clarifier parameters."""
    return {
        # 'velocity_model': 'vesilind', # 'takacs' or 'vesilind'
        'velocity_model': 'takacs', # 'takacs' or 'vesilind'
        'v0':  474.0,   # Max Vesilind settling velocity (m/day)
        'Kv':  5.76e-4, # hindered settling parameter rₕ (m^3/g)
        # Takács double-exponential extra parameters
        'v0p': 250.0,   # m/d
        'r_h': 5.76e-4, # m^3/g
        'r_p': 2.86e-3, # m^3/g
        'f_ns': 2.28e-3 # 
    }

def tss_from_cod(x13):
    """TSS (gSS/m3) per BSM1: 0.75 * sum(particulate COD)."""
    x = np.asarray(x13, float)
    return 0.75 * float(np.sum(x[particulate_cod_idx]))

def stream_compositions_from_state(y):
    """
    Given the full state vector y (5*13 + 10), return:
      eff_13, ras_13, Xe (top-layer TSS), Xu (bottom-layer TSS)
    """
    X_reactors  = y[:65].reshape(5, 13)
    X5          = X_reactors[4, :]
    y_clarifier = y[65:75]

    # soluble layers (10 x 7) follow immediately after MLSS
    Z_layers = y[75:145].reshape(10, 7)

    # BSM1 feed TSS used for scaling
    X_tss_feed = tss_from_cod(X5)

    # Clarifier layer TSS (your 10 settler states are MLSS per layer)
    Xe = float(y_clarifier[0])    # top layer = effluent TSS
    Xu = float(y_clarifier[-1])   # bottom layer = underflow TSS

    # effluent & underflow solubles from layers
    Ze_vec = Z_layers[0, :]       # top layer (effluent)
    Zu_vec = Z_layers[-1, :]      # bottom layer (underflow)

    eff_13 = map_particulates_by_tss(X5, Xe, X_tss_feed)  # solubles copied
    ras_13 = map_particulates_by_tss(X5, Xu, X_tss_feed)  # solubles copied

    # overwrite solubles with the layer values
    eff_13[soluble_idx] = Ze_vec
    ras_13[soluble_idx] = Zu_vec
    return eff_13, ras_13, Xe, Xu

def get_stabilization_influent_data():
    """
    BSM1 100-day stabilization influent (Table 5).
    Returns a function f(t)->(Q0, Zin) with constant values.
    """
    Q0 = 18446.0  # m^3/d
    Zin = np.array([
        30.00,   # S_I
        69.50,   # S_S
        51.20,   # X_I
        202.32,  # X_S
        28.17,   # X_H
        0.0,     # X_A
        0.0,     # X_P
        0.0,     # S_O
        0.0,     # S_NO
        31.56,   # S_NH
        6.95,    # S_ND
        10.59,   # X_ND
        7.00     # S_ALK (mol/m^3)
    ], dtype=float)

    def influent_data_function(t):
        return Q0, Zin

    return influent_data_function

# --- Example of how to use it ---
if __name__ == '__main__':

    # --- 1. Load Parameters and Influent Data ---
    stoich_params, Kin_params = get_bsm1_params()
    clarifier_params = get_clarifier_params()
    settling_params = get_settling_params()
    influent_file = 'influent/Inf_dry_2006.txt'
    get_influent_data = create_influent_data_function(influent_file)

    # constant influent for the 100-day stabilization (Table 5)
    get_influent_data_stab = get_stabilization_influent_data()
    print("step 1 finished")

    # --- 2. Set Initial Conditions ---
    # These are standard steady-state initial values for BSM1.
    # Each list represents one of the 5 reactors.
    initial_state_reactors = np.array([
    # S_I, S_S,   X_I,   X_S,    X_H,    X_A,   X_P, S_O,  S_NO,    S_NH,  S_ND,   X_ND, S_ALK
    [30.0, 3.7, 58.48, 23.49, 1340.0, 108.6, 178.6, 0.0, 8.41, 0.25, 0.55, 0.98, 7.0], # Tank 1
    [30.0, 1.3, 58.48,  4.27, 1340.0, 108.6, 178.6, 0.0, 5.07, 0.02, 0.35, 0.18, 7.0], # Tank 2
    [30.0, 0.5, 58.48,  1.26, 1340.0, 108.6, 178.6, 2.0, 17.6, 0.01, 0.43, 0.05, 7.0], # Tank 3
    [30.0, 0.4, 58.48,  0.81, 1340.0, 108.6, 178.6, 2.0, 18.2, 0.01, 0.41, 0.03, 7.0], # Tank 4
    [30.0, 0.3, 58.48,  0.64, 1340.0, 108.6, 178.6, 2.0, 18.4, 0.01, 0.40, 0.03, 7.0], # Tank 5
    ])
    # The ODE solver requires a flat, 1D array of initial conditions.
    # Initial conditions for the 10 clarifier layers (a reasonable gradient)
    initial_state_clarifier = np.linspace(start=50, stop=8000, num=10) # From clear top to thick bottom

    # initial conditions for the 10×7 soluble layers.
    # A simple, BSM1-compliant choice is to set all layers equal to Tank 5 solubles.
    init_Z5 = initial_state_reactors[4, soluble_idx]          # Tank 5 solubles
    initial_state_clarifier_solubles = np.tile(init_Z5, (10, 1))  # (10,7), top->bottom

    # Combine into a single flat vector for the ODE solver
    y0_init = np.concatenate([
        initial_state_reactors.flatten(),
        initial_state_clarifier,
        initial_state_clarifier_solubles.flatten()
    ])
    print("Step 2: Initial conditions defined for 145 states.")

    # --- 2b. 100-day stabilization under constant influent (no sensor noise in open-loop) ---
    # Use an implicit stiff integrator here to avoid tiny explicit Euler steps over 100 days.
    print("Starting 100-day stabilization (constant influent)...")

    def rhs_stab(t, y):
        return bsm1_plant_model(y, t, get_influent_data_stab, stoich_params, Kin_params, clarifier_params, settling_params)

    # Evaluate every 15 min (matching BSM1 KPI grid; choice of t_eval does not affect the dynamics)
    t_stab_eval = np.arange(0.0, 100.0 + 1.0/96.0, 1.0/96.0)
    sol_stab = solve_ivp(
        rhs_stab,
        t_span=(float(t_stab_eval[0]), float(t_stab_eval[-1])),
        y0=y0_init,
        t_eval=t_stab_eval,
        method='BDF',
        rtol=1e-5,
        atol=1e-7,
        max_step=1.0/24.0  # <= 1 hour steps; safe for stiff parts
    )
    if not sol_stab.success:
        print("Stabilization integrator reported:", sol_stab.message)
    y0 = sol_stab.y[:, -1]  # <-- This is now the initial state for the 14-day run
    np.save('results/y0_after_100day_stabilization.npy', y0)
    print("Stabilization finished. y0 for dynamic run saved to results/y0_after_100day_stabilization.npy")

    # ---- Table 8 steady-state (open-loop) check. Xu (underflow/RAS solids concentration) ----
    _, _, _, Xu_ss = stream_compositions_from_state(y0) # unpack only last
    print(f"Xu (underflow TSS) = {Xu_ss:.2f} g SS/m^3, Table 8 steady-state ≈ 6394 g SS/m^3")

    # --- 3. Run the Simulation ---
    print("Starting plant-wide simulation... (this may take a moment)")

    def rhs(t, y):
        return bsm1_plant_model(y, t, get_influent_data, stoich_params, Kin_params, clarifier_params, settling_params)


    # --- implicit ---   
    dt = 1.0/96.0  # implicit. 15 min in days. Similar to bsm1 output
    t_span = np.arange(0.0, 14.0 + dt, dt) 
    sol = solve_ivp(
        rhs,
        t_span=(float(t_span[0]), float(t_span[-1])),
        y0=y0,
        t_eval=t_span,
        method='BDF',              # or 'LSODA'
        rtol=1e-5,
        atol=1e-7,
        max_step=dt              # 15 min KPIs. 
    )
    if not sol.success:
        print("Integrator reported:", sol.message)
    solution = sol.y.T

    # --- Forward (Explicit) Euler on the fixed grid t_span ---
    dt = 0.00001  # explicit euler. dtsettle ≲ 0.4/250=0.0016d. 0.0001 was found to show time independene 
    t_span = np.arange(0.0, 14.0 + dt, dt) 
    solution = np.empty((t_span.size, y0.size), dtype=float)
    solution[0, :] = y0
    for k in range(t_span.size - 1):
        t  = float(t_span[k])
        dt = float(t_span[k+1] - t_span[k])
        dydt = rhs(t, solution[k, :])
        solution[k+1, :] = solution[k, :] + dt * dydt


    results_reactors  = solution[:, 0:65].reshape(len(t_span), 5, 13)
    # print("results_reactors.shape:", results_reactors.shape)

    results_clarifier = solution[:, 65:75]  # TSS profile (top->bottom or your order)
    # print("results_clarifier.shape:", results_clarifier.shape)

    # Build effluent & RAS time series from y(t)
    effluent_ts = np.empty((len(t_span), 13))
    ras_ts      = np.empty_like(effluent_ts)
    Xe_ts       = np.empty(len(t_span))
    Xu_ts       = np.empty(len(t_span))
    TSS_eff_from_cod = np.empty(len(t_span))

    for i, tt in enumerate(t_span):
        y_t = solution[i, :]
        eff_i, ras_i, Xe, Xu = stream_compositions_from_state(y_t)
        effluent_ts[i, :] = eff_i
        ras_ts[i, :]      = ras_i
        Xe_ts[i]          = Xe
        Xu_ts[i]          = Xu

        # per-step TSS from effluent states:
        TSS_eff_from_cod[i] = 0.75 * eff_i[particulate_cod_idx].sum()

        # sanity check against top-layer TSS
        if not np.isclose(TSS_eff_from_cod[i], Xe, rtol=1e-3, atol=1e-6):
            print(f"WARN t={tt:.3f}: SS_from_COD={TSS_eff_from_cod[i]:.3f} vs Xe={Xe:.3f}")

        # effluent flow
        Q0_t, _    = get_influent_data(tt)

    os.makedirs('results/across_units', exist_ok=True)

    kpis = compute_bsm1_kpis(t_span, effluent_ts, get_influent_data, clarifier_params, stoich_params)
    print("[BSM1 7–14 d | 15‑min rectangular, flow‑weighted]")
    for k, v in kpis.items():
        print(f"{k}: {v:.3f}")

    component_names = ['S_I','S_S','X_I','X_S','X_H','X_A','X_P','S_O','S_NO','S_NH','S_ND','X_ND','S_ALK']

    # Build influent concentration time series for plotting (13 columns)
    influent_concs_ts = np.vstack([get_influent_data(tt)[1] for tt in t_span])  # shape (T, 13)

    # Decide per soluble whether RAS differs from Tank 5 (i.e., solubles actually go down)
    # Always show RAS for particulates; conditionally for solubles.
    particulate_idx_full = [2, 3, 4, 5, 6, 11]  # X_I, X_S, X_H, X_A, X_P, X_ND
    RAS_PLOT_TOL = 1e-8
    solubles_go_down_by_comp = {
        j: (np.max(np.abs(ras_ts[:, j] - results_reactors[:, 4, j])) > RAS_PLOT_TOL)
        for j in soluble_idx
    }

    # across_units plotting loop
    # for comp_idx, comp_name in enumerate(component_names):
    #     fig, axes = plt.subplots(8, 1, figsize=(12, 18), sharex=True)

    #     # 1) Influent
    #     axes[0].plot(t_span, influent_concs_ts[:, comp_idx])
    #     axes[0].set_title(f'Influent → {comp_name}')

    #     # 2–6) After Tanks 1–5 (CSTR: tank conc. = tank effluent)
    #     for tank in range(5):
    #         axes[tank + 1].plot(t_span, results_reactors[:, tank, comp_idx])
    #         axes[tank + 1].set_title(f'After Tank {tank + 1} → {comp_name}')

    #     # 7) Clarifier Effluent
    #     axes[6].plot(t_span, effluent_ts[:, comp_idx])
    #     axes[6].set_title(f'Clarifier Effluent → {comp_name}')

    #     # 8) Clarifier RAS
    #     show_ras = (comp_idx in particulate_idx_full) or solubles_go_down_by_comp.get(comp_idx, False)
    #     axes[7].set_title(f'Clarifier RAS → {comp_name}')
    #     if show_ras:
    #         axes[7].plot(t_span, ras_ts[:, comp_idx])
    #     else:
    #         # If solubles don't go down, avoid plotting a misleading line
    #         axes[7].text(0.5, 0.5, 'RAS not plotted (no soluble transport detected)',
    #                     ha='center', va='center', transform=axes[7].transAxes)

    #     # Cosmetics
    #     for ax in axes:
    #         ax.grid(True)
    #         ax.set_ylabel('g/m³')
    #     axes[-1].set_xlabel('Time (days)')

    #     fig.suptitle(f'{comp_name}: Influent → Tanks 1–5 → Effluent → RAS', y=0.995)
    #     plt.tight_layout()
    #     plt.savefig(f'results/across_units/{comp_name}_across_units.png', dpi=150)
    #     plt.close()


    # Clarifier TSS figure (use Xe, Xu)
    plt.figure(figsize=(12,6))
    plt.plot(t_span, TSS_eff_from_cod, label='Effluent TSS from COD states (BSM1)')
    # Optional: overlay Xe to confirm they coincide
    plt.plot(t_span, Xe_ts, '--', label='Clarifier top-layer TSS (Xe)', alpha=0.7)
    plt.title('Plant Effluent TSS'); plt.xlabel('Time (days)'); plt.ylabel('g SS/m^3')
    plt.legend(); plt.grid(True); plt.yscale('log')
    plt.savefig('results/effluent_TSS.png'); plt.close()

    # Example 95th percentile required by BSM1:
    TSSe95 = np.percentile(TSS_eff_from_cod, 95)
    print("TSSe95 =", TSSe95)

    # make the validation figure + CSV
    make_bsm1_validation_plot(kpis, TSSe95)