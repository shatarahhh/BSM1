# clarifier_model.py File

import numpy as np

def vesilind_settling_velocity(X, settling_params):
    """
    Calculates settling velocity using the Vesilind single-exponential model.

    Args:
        X (float or array): Suspended solids concentration (g/m^3).
        settling_params (dict): Dictionary of clarifier parameters containing 'v0' and 'Kv'.

    Returns:
        float or array: The settling velocity (m/day).
    """
    v0 = settling_params['v0']  # Max settling velocity (m/day)
    Kv = settling_params['Kv']           # Vesilind settling parameter (m^3/g)
    
    # Ensure concentration is non-negative for the calculation
    X_safe = np.maximum(0.0, np.asarray(X, float)) # type: ignore
    
    v_s = v0 * np.exp(-Kv * X_safe)
    return v_s

def takacs_settling_velocity(X, settling_params):
    """
    Takács double-exponential with non-settleable fraction f_ns (BSM1, p.9).
    Uses (X_eff = (1 - f_ns)*X) in the hindered/flocculant terms.
    """
    v0  = settling_params['v0']   # m/d  (Vesilind max term used in double exp)
    v0p = settling_params['v0p']  # m/d  (additional max term)
    r_h = settling_params['r_h']  # m^3/g (hindered)
    r_p = settling_params['r_p']  # m^3/g (flocculant)
    fns = settling_params['f_ns'] # dimensionless

    X = np.maximum(0.0, np.asarray(X, float)) # type: ignore
    X_eff = (1.0 - fns) * X
    # Commonly used form consistent with BSM1 data: vs(X) = v0 * exp(-r_h * X_eff) + v0p * exp(-r_p * X_eff)
    return v0 * np.exp(-r_h * X_eff) + v0p * np.exp(-r_p * X_eff)

def takacs_clarifier_model(X_layers, X_in_total, Q_in, clarifier_params, settling_params):
    """
    Implements the BSM1 1D 10-layer secondary clarifier (Takács) with the
    clarification/thickening split and the X_t = 3000 g/m^3 switch in the
    clarification flux J_clar (BSM1, Section 2.3.3, pp. 9–11).

    Args:
        X_layers (10,) array: TSS profile from top (index 0) to bottom (index 9).
        X_in_total (float): Total particulate conc. from the last reactor (g/m^3).
        Q_in (float): Clarifier feed flow (m^3/day) = Q_e + Q_u.
        clarifier_params (dict): {'A','N_layers','h','Q_RAS','Q_w','feed_layer','X_t'}.
        settling_params (dict): settling velocity parameters + 'velocity_model'.

    Returns:
        tuple: (dxdt (10,), X_underflow_total, X_effluent_total)
    """
    # --- Unpack geometry & hydraulics ---
    A          = clarifier_params['A']          # m^2
    N_layers   = clarifier_params['N_layers']   # 10
    h          = clarifier_params['h']          # m (layer height)
    Q_RAS      = clarifier_params['Q_RAS']      # m^3/d
    Q_w        = clarifier_params['Q_w']        # m^3/d
    feed_layer = clarifier_params['feed_layer'] # 0-based; top=0, bottom=9
    X_t        = clarifier_params.get('X_t', 3000.0)  # g/m^3

    Q_u  = Q_RAS + Q_w               # underflow (m^3/d)
    Q_e  = Q_in - Q_u                # effluent (m^3/d)
    v_up = Q_e / A                   # m/d (upward, clarification)
    v_dn = Q_u / A                   # m/d (downward, thickening)

    # --- States and settling velocities ---
    X = np.maximum(0.0, np.asarray(X_layers, dtype=float)) # type: ignore
    model = settling_params.get('velocity_model', 'takacs').lower()
    if model == 'vesilind':
        vs = vesilind_settling_velocity(X, settling_params)
    else:
        vs = takacs_settling_velocity(X, settling_params)

    # --- Gravitational flux at interfaces (j = 1..N_layers-1) ---
    # Interface j sits between upper layer (j-1) and lower layer (j).
    # Above feed (j <= feed_layer): use J_clar with the X_t switch.
    # Below feed (j > feed_layer): use J_s^min (hindered/flocculant min-flux).
    Jg = np.zeros(N_layers)  # g/(m^2·d); Jg[1..N_layers-1] used
    for j in range(1, N_layers): # Jg[0] is never used.
        upper, lower = j - 1, j
        if j <= feed_layer:
            # Clarification flux with threshold X_t (BSM1 definition)
            J_down = vs[lower] * X[lower]
            if X[lower] <= X_t:
                Jg[j] = J_down
            else:
                Jg[j] = min(J_down, vs[upper] * X[upper])
        else:
            # Thickening: limiting settling flux (min supply/capacity)
            Jg[j] = min(vs[upper] * X[upper], vs[lower] * X[lower])

    # --- Layer-wise mass balances (bulk advection + gravitational flux divergence) ---
    V_layer = A * h
    A_vup   = A * v_up
    A_vdn   = A * v_dn
    dxdt    = np.zeros(N_layers, dtype=float)

    for i in range(N_layers):  # i: 0=top, 9=bottom
        mass_in  = 0.0
        mass_out = 0.0

        # Upward bulk advection (clarification side)
        if i == 0:
            # Top layer: only inflow from below by upflow; outflow via J_clar (not v_up*X_top)
            if N_layers > 1:
                mass_in += A_vup * X[1]
        elif i < feed_layer:
            # Clarification zone (interior): v_up*(X_{i+1} - X_i)
            mass_in  += A_vup * X[i + 1]
            mass_out += A_vup * X[i]
        elif i == feed_layer:
            # Feed layer: upflow leaves through its upper interface
            mass_out += A_vup * X[i]
        # below feed: no upward bulk advection

        # Downward bulk advection (thickening side)
        if i > feed_layer:
            # Thickening zone (interior): v_dn*(X_{i-1} - X_i)
            mass_in  += A_vdn * X[i - 1]
            mass_out += A_vdn * X[i]
        elif i == feed_layer:
            # Feed layer: part of feed leaves downward
            mass_out += A_vdn * X[i]
        # above feed: no downward bulk advection

        # External feed into the feed layer (Q_f * X_f)
        if i == feed_layer:
            mass_in += Q_in * X_in_total

        # Gravitational flux divergence
        if i == 0:
            grav_in  = 0.0
            grav_out = A * Jg[1]                    # -J_clar,10 in BSM1
        elif i == N_layers - 1:
            grav_in  = A * Jg[N_layers - 1]         # +J_s at interface above bottom
            grav_out = 0.0
        else:
            grav_in  = A * Jg[i]                    # from interface (i-1 | i)
            grav_out = A * Jg[i + 1]                # to interface (i | i+1)

        net_mass_change = (mass_in + grav_in) - (mass_out + grav_out)
        dxdt[i] = net_mass_change / V_layer

    # Outputs
    X_underflow_total = X[-1]   # bottom layer
    X_effluent_total  = X[0]    # top layer
    return dxdt, X_underflow_total, X_effluent_total