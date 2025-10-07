# clarifier_model.py File

import numpy as np

def vesilind_settling_velocity(X, settling_params):
    """
    Calculates settling velocity using the Vesilind single-exponential model.

    Args:
        X (float or array): Suspended solids concentration (g/m^3).
        settling_params (dict): Dictionary of clarifier parameters containing 'v0_vesilind' and 'Kv'.

    Returns:
        float or array: The settling velocity (m/day).
    """
    v0 = settling_params['v0_vesilind']  # Max settling velocity (m/day)
    Kv = settling_params['Kv']           # Vesilind settling parameter (m^3/g)
    
    # Ensure concentration is non-negative for the calculation
    X_safe = np.maximum(0, X)
    
    v_s = v0 * np.exp(-Kv * X_safe)
    return v_s

def takacs_clarifier_model(X_clarifier, X_in_total, Q_in, clarifier_params, settling_params):
    """
    Implements the 1D Takács dynamic model for a 10-layer secondary clarifier.

    Args:
        X_clarifier (array): Current solids concentrations in the 10 layers (g/m^3).
        X_in_total (float): Total particulate concentration from the last reactor (g/m^3).
        Q_in (float): Flow rate from the last reactor (m^3/day).
        clarifier_params (dict): Dictionary of clarifier parameters.

    Returns:
        tuple: (
            dxdt (array): The derivatives for the 10 layer concentrations.
            X_underflow_total (array)
            X_effluent_total (array)
        )
    """
    # Unpack parameters
    A = clarifier_params['A']              # Clarifier area (m^2)
    N_layers = clarifier_params['N_layers']# Number of layers
    h = clarifier_params['h']              # Height of each layer (m)
    Q_RAS = clarifier_params['Q_RAS']      # Return Activated Sludge (underflow) rate (m^3/day)
    Q_w = clarifier_params['Q_w']      # Return Activated Sludge (underflow) rate (m^3/day)
    feed_layer = clarifier_params['feed_layer'] # Index of the feed layer (e.g., 4 for the 5th layer)

    Q_u = Q_RAS + Q_w          # Underflow rate (m^3/day)
    # Calculate hydraulic velocities
    Q_eff = Q_in - Q_u
    v_up = Q_eff / A          # Upflow velocity (overflow)
    v_down = Q_u / A        # Downflow velocity (underflow)

    # Initialize arrays for fluxes
    J_settling = np.zeros(N_layers + 1)
    J_bulk_up = np.zeros(N_layers + 1)
    J_bulk_down = np.zeros(N_layers + 1)
    dxdt = np.zeros(N_layers)

    # Calculate settling velocity for each layer's concentration
    v_s = vesilind_settling_velocity(X_clarifier, settling_params)

    # --- Calculate Solids Fluxes at layer interfaces ---
    # J(i) is the flux between layer i-1 and i
    # Initialize the settling flux for the top boundary to zero.
    J_settling[0] = 0

    # Loop through the remaining interfaces (from layer 1 to the bottom).
    for i in range(1, N_layers):
        # The settling flux is always the minimum of what the layer above can supply and what the layer below can accept.
        flux_from_above = v_s[i-1] * X_clarifier[i-1]
        flux_capacity_below = v_s[i] * X_clarifier[i]
        
        J_settling[i] = min(flux_from_above, flux_capacity_below)

    # --- Mass Balance for Each Layer ---
    V_layer = A * h
    
    for i in range(N_layers):
    # Calculate solids moving INTO the current layer 'i'
    # FLUX IN from the layer above (i-1)
        if i == 0:
            # Top layer (i=0) has no layer above it.
            flux_down_from_above = 0
        else:
            # This is the sum of solids carried by bulk water flow (v_down) and the pre-calculated limiting gravitational flux (J_settling).
            flux_down_from_above = (A * v_down * X_clarifier[i-1]) + (A * J_settling[i])

        # FLUX IN from the layer below (i+1)
        if i == N_layers - 1:
            # Bottom layer (i=9) has no layer below it.
            flux_up_from_below = 0
        else:
            # Solids are carried up by the bulk effluent flow (v_up).
            flux_up_from_below = A * v_up * X_clarifier[i+1]

        # FLUX IN from the feed pipe
        if i == feed_layer:
            # The designated feed layer receives solids from the bioreactor.
            feed_flux = Q_in * X_in_total
        else:
            feed_flux = 0

        # Calculate solids moving OUT OF the current layer 'i'
        # FLUX OUT to the layer above (i-1)
        # Solids are carried upwards by the effluent bulk flow.
        flux_up_to_above = A * v_up * X_clarifier[i]

        # FLUX OUT to the layer below (i+1)
        if i == N_layers - 1:
            # For the bottom layer, "out" means being removed in the RAS and waste flow (Q_u).
            # There is no settling "out" of the very bottom.
            flux_down_to_below = Q_u * X_clarifier[i]
        else:
            # This is the sum of solids carried by bulk water flow and the
            # pre-calculated limiting gravitational flux leaving layer 'i' for 'i+1'.
            flux_down_to_below = (A * v_down * X_clarifier[i]) + (A * J_settling[i+1])

        # Calculate the final rate of change
        # The change in mass is (everything IN) - (everything OUT)
        net_mass_change_rate = (flux_down_from_above + flux_up_from_below + feed_flux) - \
                            (flux_down_to_below + flux_up_to_above)

        # The rate of change in concentration is the rate of mass change divided by the layer volume.
        dxdt[i] = net_mass_change_rate / V_layer

    # Calculate output concentrations
    X_underflow_total = X_clarifier[-1]
    X_effluent_total = X_clarifier[0]

    # Distribute the total particulate concentration back to individual components

    return dxdt, X_underflow_total, X_effluent_total