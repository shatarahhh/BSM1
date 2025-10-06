import numpy as np
from ASM1_Processes import calculate_process_rates
from clarifier_model import takacs_clarifier_model


def bsm1_plant_model(y, t, influent_data, stoich_params, Kin_params, clarifier_params, settling_params):
    """
    The main function describing the BSM1 plant dynamics.
    This function will be passed to an ODE solver.

    Args:
        y (array): A flat 1D array of 65 elements (5 tanks * 13 components)
                   representing the current state of the system.
        t (float): The current time step (required by the ODE solver).
        influent_data (function): A function that returns the influent concentrations
                                  and flow rate for a given time 't'.
        params (dict): A dictionary of model parameters.

    Returns:
        array: A flat 1D array of the 65 derivatives (dC/dt).
    """

# --- Unpack State Vector ---
    # First 65 elements are for the 5 bioreactors
    y_reactors = y[0:65]
    current_state_matrix = y_reactors.reshape(5, 13)
    # Last 10 elements are for the 10 clarifier layers
    y_clarifier = y[65:75]

    # --- Define Stoichiometric Parameters ---
    # These values are taken from the BSM1 report, Table 3.
    Y_A = stoich_params['Y_A'] # g COD/g N  (Yield for autotrophs)
    Y_H = stoich_params['Y_H'] # g COD/g COD (Yield for heterotrophs)
    f_P = stoich_params['f_P'] # dimensionless (Fraction of biomass yielding particulate products)
    i_XB = stoich_params['i_XB']  # g N/g COD  (Mass of N in biomass)
    i_XP = stoich_params['i_XP'] # g N/g COD  (Mass of N in particulate products)

    # --- Build the Stoichiometric Matrix with Parameters ---
    # This matrix is now readable and directly maps to the ASM1 documentation.
    # Rows: 13 components (S_I, S_S, ..., S_ALK)
    # Columns: 8 processes (rho_1, ..., rho_8)
    # The matrix is transposed (.T) to align with the shape of the process rates vector,

    stoichiometric_matrix = np.array([
        # Component: S_I, S_S, X_I, X_S, X_H, X_A, X_P, S_O, S_NO, S_NH, S_ND, X_ND, S_ALK
        [0, -1/Y_H, 0,     0,  1,  0,   0,    -(1-Y_H)/Y_H,                     0,         -i_XB,  0,  0,                 -i_XB/14],  # Process 1: Aerobic growth, Heterotrophs
        [0, -1/Y_H, 0,     0,  1,  0,   0,               0, -((1-Y_H)/(2.86*Y_H)),         -i_XB,  0,  0, -((1-Y_H)/(2.86*Y_H))/14],  # Process 2: Anoxic growth, Heterotrophs
        [0,      0, 0, 1-f_P, -1,  0, f_P,               0,                     0,             0,  0,  0,                        0], # Process 3: Decay, Heterotrophs (Note: N is handled in process 8)
        [0,      0, 0,     0,  0,  1,   0, -(4.57-Y_A)/Y_A,                 1/Y_A, -1/Y_A - i_XB,  0,  0,     -i_XB/14 - 1/(7*Y_A)],  # Process 4: Aerobic growth, Autotrophs
        [0,      0, 0, 1-f_P,  0, -1, f_P,               0,                     0,             0,  0,  0,                        0], # Process 5: Decay, Autotrophs (Note: N is handled in process 8)
        [0,      0, 0,     0,  0,  0,   0,               0,                     0,             1, -1,  0,                     1/14],    # Process 6: Ammonification
        [0,      1, 0,    -1,  0,  0,   0,               0,                     0,             0,  0,  0,                        0],       # Process 7: Hydrolysis
        [0,      0, 0,     0,  0,  0,   0,               0,                     0,             0,  1, -1,                        0],  # Process 8: Hydrolysis of N. (This represents the release of N from decay, linking it to hydrolysis rate)
    ]).T
    # --- 1. Reshape the State Vector ---
    # Convert the flat 'y' array into a 5x13 matrix for easier handling.
    # Each row represents a tank, each column a component.

    # --- 2. Get Influent Conditions for the current time 't' ---
    influent_flow, influent_concs = influent_data(t)

    # --- 3. Define BSM1 Plant Parameters & Flows ---
    V = [1000, 1000, 1333, 1333, 1333] # Volumes of the 5 reactors (m^3)
    Q_RAS = clarifier_params['Q_RAS']  # Return Activated Sludge flow rate (m^3/day)
    Q_IR = 55338   # Internal Recirculation flow rate (m^3/day)

    # --- 4. Initialize the Derivative Array ---
    dydt = np.zeros_like(current_state_matrix)

    # --- 5. Loop Through Each Reactor to Calculate Mass Balance ---
    for i in range(5): # i = 0 to 4, representing Tank 1 to 5
        
        # Create the 'state' dictionary for the current reactor
        state_dict = {
            'S_I': current_state_matrix[i, 0], 
            'S_S': current_state_matrix[i, 1],
            'X_I': current_state_matrix[i, 2], 
            'X_S': current_state_matrix[i, 3],
            'X_H': current_state_matrix[i, 4], 
            'X_A': current_state_matrix[i, 5],
            'X_P': current_state_matrix[i, 6], 
            'S_O': current_state_matrix[i, 7],
            'S_NO': current_state_matrix[i, 8], 
            'S_NH': current_state_matrix[i, 9],
            'S_ND': current_state_matrix[i, 10], 
            'X_ND': current_state_matrix[i, 11],
            'S_ALK': current_state_matrix[i, 12]
        }

        # Set Oxygen concentration (S_O) based on tank type
        if i < 2: # Anoxic tanks (Tank 1 and 2)
            state_dict['S_O'] = 0
        else: # Aerobic tanks (Tank 3, 4, 5)
            # In a real simulation, this would be controlled. We'll fix it for now.
            state_dict['S_O'] = 2.0

        # --- Calculate Biological Reaction Rates (The "Engine") ---
        process_rates = np.array(calculate_process_rates(state_dict, Kin_params))

        # --- Calculate Net Reaction Term (r_C) for all 13 components ---
        # This is the dot product of the stoichiometric matrix and the process rates vector.
        r_C = stoichiometric_matrix.dot(process_rates)

        # --- Calculate Transport Term (Flow In - Flow Out) ---
        inflow_rate = influent_flow + Q_IR + Q_RAS 
        if i == 0: # Tank 1
            flow_in = ((influent_flow * influent_concs            ) + 
                       (Q_IR          * current_state_matrix[4, :]) + # from Tank 5
                       (Q_RAS         * current_state_matrix[4, :]))  # Wrong simplification. It should be from settled sludge from secondary clarifier after tank 5
            
            flow_out = inflow_rate * current_state_matrix[i, :]
        else: # Tanks 2 to 5
            flow_in  = inflow_rate * current_state_matrix[i-1, :] # from previous tank
            flow_out = inflow_rate * current_state_matrix[i, :]

        # --- Calculate the Final Derivative (dC/dt) ---
        dydt[i, :] = (flow_in - flow_out) / V[i] + r_C

    # --- 6. Flatten the Derivative Matrix ---
    # The ODE solver requires a flat 1D array as output.
    return dydt.flatten()