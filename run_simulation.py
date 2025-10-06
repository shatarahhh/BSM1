# In file: run_simulation.py

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os

from bsm1_simulation import bsm1_plant_model, plant_wide_model
from ASM1_Processes import calculate_process_rates

def get_bsm1_params():
    """
    Returns a dictionary containing all the kinetic and stoichiometric
    parameters for the BSM1 model, corrected for a temperature of 15 deg C.
    """
    
    # Note: The original BSM1 parameters are defined at 20 deg C.
    # We must apply temperature correction for key kinetic parameters.
    # The Arrhenius equation is used: K_T = K_20 * exp(theta * (T - 20))
    # For BSM1, the standard temperature is T = 15 deg C.
    
    T = 15  # Operating temperature
    T_ref = 20 # Reference temperature
    
    # Kinetic parameters at 20 deg C (from BSM1 doc Table 4)
    mu_H_20 = 4.0   # per day
    b_H_20 = 0.3    # per day
    k_h_20 = 3.0    # per day
    k_a_20 = 0.05   # m^3/(g N * day)
    mu_A_20 = 0.5   # per day
    b_A_20 = 0.05   # per day

    # Temperature correction using Arrhenius equation
    mu_H = mu_H_20 * np.exp(0.0693 * (T - T_ref))
    b_H = b_H_20 * np.exp(0.0693 * (T - T_ref))
    k_h = k_h_20 * np.exp(0.0693 * (T - T_ref))
    k_a = k_a_20 * np.exp(0.0693 * (T - T_ref))
    mu_A = mu_A_20 * np.exp(0.0693 * (T - T_ref))
    b_A = b_A_20 * np.exp(0.0693 * (T - T_ref))
    
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
        'mu_H': mu_H,     # Max specific growth rate for heterotrophs
        'K_S': 10.0,      # Substrate half-saturation constant for heterotrophs (g COD/m^3)
        'K_O_H': 0.2,     # Oxygen half-saturation constant for heterotrophs (g -COD/m^3)
        'K_NO': 0.5,      # Nitrate half-saturation constant for heterotrophs (g N/m^3)
        'b_H': b_H,       # Decay rate for heterotrophs
        'eta_g': 0.8,     # Correction factor for anoxic growth of heterotrophs
        'eta_h': 0.6,     # Correction factor for anoxic hydrolysis
        'k_h': k_h,       # Hydrolysis rate constant
        'K_X': 0.1,       # Particulate substrate half-saturation constant (g COD/g COD)
        'mu_A': mu_A,     # Max specific growth rate for autotrophs
        'K_NH_A': 1.0,    # Ammonia half-saturation constant for autotrophs (g N/m^3)
        'K_O_A': 0.4,     # Oxygen half-saturation constant for autotrophs (g -COD/m^3)
        'b_A': b_A,       # Decay rate for autotrophs
        'k_a': k_a,       # Ammonification rate constant
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
                                 bounds_error=False, fill_value=(flow_rate_values[0], flow_rate_values[-1]))
    
    # Create an interpolation function for the 13 concentrations
    conc_interpolator = interp1d(time_points, concentration_values, kind='linear', axis=0,
                                 bounds_error=False, fill_value=(concentration_values[0], concentration_values[-1]))

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
        'feed_layer': 4,    # Feed enters the 5th layer (0-indexed)
    }

def get_settling_params():
    """Returns a dictionary of the Takács clarifier parameters."""
    return {
        'v0_vesilind':  474.0,   # Max Vesilind settling velocity (m/day)
        'Kv':           2.86e-3, # Flocculant zone settling parameter (m^3/g)
    }
# is that kv correcT?

# --- Example of how to use it ---
if __name__ == '__main__':

    # --- 1. Load Parameters and Influent Data ---
    stoich_params, Kin_params = get_bsm1_params()
    clarifier_params = get_clarifier_params()
    settling_params = get_settling_params()
    influent_file = 'influent/Inf_dry_2006.txt'
    get_influent_data = create_influent_data_function(influent_file)
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

    # Combine into a single flat vector for the ODE solver
    y0 = np.concatenate([initial_state_reactors.flatten(), initial_state_clarifier])
    print("Step 2: Initial conditions defined for 75 states.")

    # --- 3. Define Simulation Time ---
    # Simulate for 14 days, with output every 0.1 days.
    t_span = np.arange(0, 14, 0.1)

    # --- 4. Run the Simulation ---
    print("Starting plant-wide simulation... (this may take a moment)")
    solution = odeint(bsm1_plant_model, y0, t_span, 
                      args=(get_influent_data, stoich_params, Kin_params, clarifier_params, settling_params))
    print("Simulation finished successfully!")
    # --- 5. Process and Plot Results ---
    # Reshape the solution back into a 3D matrix: (time, tank, component)
    # Reshape reactor results: (time, tank, component)
    results_reactors = solution[:, 0:65].reshape(len(t_span), 5, 13)
    # Get clarifier results: (time, layer)
    results_clarifier = solution[:, 65:75]

    # Define the names of the components for plot titles and filenames
    component_names = ['S_I', 'S_S', 'X_I', 'X_S', 'X_H', 'X_A', 'X_P',
                    'S_O', 'S_NO', 'S_NH', 'S_ND', 'X_ND', 'S_ALK']

    # Create a directory for results if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    print("\nGenerating and saving reactor plots...")

    for i in range(13):
        plt.figure(figsize=(12, 6))
        plt.plot(t_span, results_reactors[:, 4, i], label=f'{component_names[i]} in Effluent (Tank 5)')
        plt.title(f'Bioreactor Effluent: {component_names[i]} Concentration')
        plt.xlabel('Time (days)'); plt.ylabel(f'Concentration (g/m^3)')
        plt.legend(); plt.grid(True)
        plt.savefig(f'results/reactor_{component_names[i]}.png')
        plt.close()
    
    print("Generating and saving clarifier plots...")
    
    # Plot Effluent and Underflow TSS from clarifier
    effluent_TSS = results_clarifier[:, 0]
    underflow_TSS = results_clarifier[:, -1]
    
    plt.figure(figsize=(12, 6))
    plt.plot(t_span, effluent_TSS, label='Effluent TSS (Top Layer)')
    plt.plot(t_span, underflow_TSS, label='Underflow TSS (Bottom Layer)')
    plt.title('Clarifier Performance: Total Suspended Solids (TSS)')
    plt.xlabel('Time (days)'); plt.ylabel('TSS Concentration (g/m^3)')
    plt.legend(); plt.grid(True); plt.yscale('log')
    plt.savefig('results/clarifier_TSS.png')
    plt.close()

    print("\nAll plots have been saved to the 'results' folder.")

'''
    # Loop through each of the 13 components
    for i in range(13):
        # Extract the data for the current component from the final tank (index 4)
        component_effluent = results[:, 4, i]
        # Get the name of the component for labeling
        current_component_name = component_names[i]
        # Create a new figure for each plot
        plt.figure(figsize=(12, 6))
        # Plot the data
        plt.plot(t_span, component_effluent, label=f'{current_component_name} in Effluent (Tank 5)')
        # Add titles and labels
        plt.title(f'BSM1 Simulation: {current_component_name} Concentration')
        plt.xlabel('Time (days)')
        plt.ylabel(f'{current_component_name} Concentration (g/m^3)')
        plt.legend()
        plt.grid(True)
        # Save the plot to a unique file
        file_name = f'results_{current_component_name}.png'
        plt.savefig(file_name)
        # Close the figure to free up memory
        plt.close()s

    print("All plots have been saved successfully.")
'''