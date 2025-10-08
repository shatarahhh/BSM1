# ASM1_Processes.py File

def calculate_process_rates(state, Kin_params):
    """
    Calculates the 8 biological process rates for ASM1 based on the BSM1 standard.

    Args:
        state (dict): A dictionary of the current concentrations of the 13 state variables.
                      Keys are the component names (e.g., 'S_S', 'X_H').
        kin_params (dict): A dictionary of the model's kinetic parameters.
                       Keys are the parameter names (e.g., 'mu_H', 'K_S').

    Returns:
        list: A list containing the 8 calculated process rates [rho_1, ..., rho_8].
    """

    # Unpack state variables for easier use
    S_I = state['S_I']   # Inert soluble organic matter
    S_S = state['S_S']   # Readily biodegradable substrate
    X_I = state['X_I']   # Inert particulate organic matter
    X_S = state['X_S']   # Slowly biodegradable substrate
    X_H = state['X_H']   # Active heterotrophic biomass
    X_A = state['X_A']   # Active autotrophic biomass
    X_P = state['X_P']   # Particulate products from biomass decay
    S_O = state['S_O']   # Oxygen
    S_NO = state['S_NO'] # Nitrate and nitrite nitrogen
    S_NH = state['S_NH'] # Ammonia nitrogen
    S_ND = state['S_ND'] # Soluble biodegradable organic nitrogen
    X_ND = state['X_ND'] # Particulate biodegradable organic nitrogen
    S_ALK = state['S_ALK'] # Alkalinity

    # Unpack parameters for easier use
    mu_H = Kin_params['mu_H']
    K_S = Kin_params['K_S']
    K_O_H = Kin_params['K_O_H']
    K_NO = Kin_params['K_NO']
    b_H = Kin_params['b_H']
    eta_g = Kin_params['eta_g']
    eta_h = Kin_params['eta_h']
    k_h = Kin_params['k_h']
    K_X = Kin_params['K_X']
    mu_A = Kin_params['mu_A']
    K_NH_A = Kin_params['K_NH_A'] # Note: BSM1 uses K_NH for autotrophs, but ASM1 matrix shows K_NH,A
    K_O_A = Kin_params['K_O_A']
    b_A = Kin_params['b_A']
    k_a = Kin_params['k_a']

    # --- Process Rate Calculations ---

    # Process 1: Aerobic growth of heterotrophs
    # This is the consumption of organic matter (S_S) by heterotrophs in the presence of oxygen.
    rho_1 = mu_H * (S_S / (K_S + S_S)) * (S_O / (K_O_H + S_O)) * X_H

    # Process 2: Anoxic growth of heterotrophs (Denitrification)
    # This is the consumption of organic matter (S_S) using nitrate (S_NO) instead of oxygen.
    rho_2 = mu_H * (S_S / (K_S + S_S)) * (K_O_H / (K_O_H + S_O)) * (S_NO / (K_NO + S_NO)) * eta_g * X_H

    # Process 3: Decay of heterotrophs
    # This represents the death of heterotrophic bacteria.
    rho_3 = b_H * X_H

    # Process 4: Aerobic growth of autotrophs (Nitrification)
    # This is the conversion of ammonia (S_NH) to nitrate (S_NO) by autotrophs.
    rho_4 = mu_A * (S_NH / (K_NH_A + S_NH)) * (S_O / (K_O_A + S_O)) * X_A

    # Process 5: Decay of autotrophs
    # This represents the death of autotrophic bacteria.
    rho_5 = b_A * X_A

    # Process 6: Ammonification
    # The breakdown of soluble organic nitrogen (S_ND) into ammonia (S_NH).
    rho_6 = k_a * S_ND * X_H

    # Process 7: Hydrolysis of entrapped organics
    # The breakdown of complex, slowly biodegradable organics (X_S) into a form (S_S) that bacteria can eat.
    # This occurs under both aerobic and anoxic conditions.
    term_hydrolysis_switch = (S_O / (K_O_H + S_O)) + eta_h * (K_O_H / (K_O_H + S_O)) * (S_NO / (K_NO + S_NO))
    
    rho_7 = k_h * (X_S / (K_X * X_H + X_S)) * term_hydrolysis_switch * X_H

    # Process 8: Hydrolysis of entrapped organic nitrogen
    # This process is biochemically linked to Process 7. It breaks down particulate organic nitrogen (X_ND).
    # The rate is derived from the hydrolysis of organics.
    
    rho_8 = k_h * (X_ND / (K_X * X_H + X_S)) * term_hydrolysis_switch * X_H 

    return [rho_1, rho_2, rho_3, rho_4, rho_5, rho_6, rho_7, rho_8]