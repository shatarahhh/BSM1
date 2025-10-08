# bsm1_simulation.py File

import numpy as np
from ASM1_Processes import calculate_process_rates
from clarifier_model import takacs_clarifier_model  # clarifier ODE (10 layers)

# indices: [2..6] are particulate COD; 11 is X_ND (particulate N)
particulate_cod_idx  = [2, 3, 4, 5, 6] # X_I, X_S, X_H, X_A, X_P
all_particulate_idx  = [2, 3, 4, 5, 6, 11] # add X_ND

def map_particulates_by_tss(x_src_13, X_tss_target, MLSS_in_clarifier):
    """
    Keep particulate *fractions* identical to source x_src_13, but
    scale their absolute values so that TSS equals X_tss_target.
    Soluble components are copied unchanged.
    """
    x_out = np.copy(x_src_13)
    ratio = float(X_tss_target / (MLSS_in_clarifier + 1e-12))
    x_out[all_particulate_idx] = ratio * x_src_13[all_particulate_idx]
    return x_out

def bsm1_plant_model(y, t, influent_data, stoich_params, Kin_params, clarifier_params, settling_params):
    """
    BSM1 plant dynamics (5 reactors + 10-layer clarifier).
    y has 75 states: 5*13 (reactors) + 10 (clarifier layers, MLSS).
    """

    # --- 1) Unpack state vector ---
    y_reactors   = y[0:65]
    X_reactors   = y_reactors.reshape(5, 13)     # 5 tanks x 13 ASM1 states
    y_clarifier  = y[65:75]                      # 10-layer MLSS profile (bottom->top or your chosen convention)

    # --- 2) Stoichiometry (unchanged) ---
    Y_A = stoich_params['Y_A']
    Y_H = stoich_params['Y_H']
    f_P = stoich_params['f_P']
    i_XB = stoich_params['i_XB']
    i_XP = stoich_params['i_XP']

    stoichiometric_matrix = np.array([
        # S_I,   S_S,      X_I,    X_S,        X_H,  X_A,  X_P,    S_O,                     S_NO,                      S_NH,                 S_ND,                 X_ND,                               S_ALK
        [0,    -1/Y_H,     0,      0,          1,    0,    0,   -(1-Y_H)/Y_H,               0,                        -i_XB,                 0,                    0,                                  -i_XB/14],  # ρ1
        [0,    -1/Y_H,     0,      0,          1,    0,    0,    0,                        -((1-Y_H)/(2.86*Y_H)),    -i_XB,                 0,                    0,       ((1-Y_H)/(2.86*Y_H))/14 - i_XB/14],  # ρ2   (fixed)
        [0,       0,       0,    1-f_P,       -1,    0,   f_P,   0,                         0,                         0,                    0,      i_XB - f_P*i_XP,                         0],  # ρ3   (fixed)
        [0,       0,       0,      0,          0,    1,    0,   -(4.57-Y_A)/Y_A,            1/Y_A,             -1/Y_A - i_XB,              0,                    0,                 -i_XB/14 - 1/(7*Y_A)],  # ρ4
        [0,       0,       0,    1-f_P,        0,   -1,   f_P,   0,                         0,                         0,                    0,      i_XB - f_P*i_XP,                         0],  # ρ5   (fixed)
        [0,       0,       0,      0,          0,    0,    0,   0,                          0,                         1,                   -1,                    0,                                 1/14],  # ρ6
        [0,       1,       0,     -1,          0,    0,    0,   0,                          0,                         0,                    0,                    0,                                  0],  # ρ7
        [0,       0,       0,      0,          0,    0,    0,   0,                          0,                         0,                    1,                   -1,                                  0],  # ρ8
    ]).T

    # --- 3) Influent & flows ---
    influent_flow, influent_concs = influent_data(t)  # Q0, Zin (13,)
    V_reactors = [1000, 1000, 1333, 1333, 1333]
    Q_RAS = clarifier_params['Q_RAS']  # return sludge (underflow to Tank 1)
    Q_w   = clarifier_params['Q_w']          #  wastage
    Q_IR  = 55338                      # internal recycle (Tank5 -> Tank1)

    # Reactor through-flow is constant in all tanks:
    # Q_reac = Q0 + Qa + Qr
    outflow_rate = influent_flow + Q_IR + Q_RAS

    # --- 4) Clarifier feed & MLSS  ---
    # MLSS entering clarifier = 0.75 * (sum of particulate COD in Tank 5)
    total_particulate_cod_t5 = float(np.sum(X_reactors[4, particulate_cod_idx]))
    MLSS_in_clarifier = 0.75 * total_particulate_cod_t5

    # Clarifier feed flow (BSM1): Qf = Q5 - Qa = Q0 + Qr
    Q_to_clarifier = influent_flow + Q_RAS  # do NOT add Q_IR

    # --- 5) Run clarifier ODE (pass Vesilind params) (FIXED CALL) ---
    # Provide minimal defaults if not present

    clar_dxdt, X_u_total, X_e_total = takacs_clarifier_model(
        y_clarifier,
        MLSS_in_clarifier,
        Q_to_clarifier,
        clarifier_params,
        settling_params
    )

    # --- 6) Build RAS composition by MLSS ratio (particulates scaled, solubles copied) ---
    RAS_concs_full = map_particulates_by_tss(X_reactors[4, :], X_u_total, MLSS_in_clarifier)

    # --- 7) Reactor mass balances ---
    dydt_reactors = np.zeros_like(X_reactors)

    for i in range(5):
        state_dict = dict(zip(
            ['S_I','S_S','X_I','X_S','X_H','X_A','X_P','S_O','S_NO','S_NH','S_ND','X_ND','S_ALK'],
            X_reactors[i, :]
        ))

        rhos = np.array(calculate_process_rates(state_dict, Kin_params))
        r_C  = stoichiometric_matrix.dot(rhos) # type: ignore

        if i == 0:
            flow_in  = (influent_flow * influent_concs) + \
                       (Q_IR * X_reactors[4, :])        + \
                       (Q_RAS * RAS_concs_full)
        else:
            flow_in  = outflow_rate * X_reactors[i-1, :]

        flow_out = outflow_rate * X_reactors[i, :]

        dydt_reactors[i, :] = (flow_in - flow_out) / V_reactors[i] + r_C

        S_O_idx = 7
        S_O_star = 8.0
        KLa = [0.0, 0.0, 240.0, 240.0, 84.0]  # default open-loop; for closed-loop let KLa[4] vary
        dydt_reactors[i, S_O_idx] += KLa[i] * (S_O_star - X_reactors[i, S_O_idx])

    # print("dydt_reactors shape:", dydt_reactors.shape)
    # print("dydt_reactors.flatten() shape:", dydt_reactors.flatten().shape)
    # print("clar_dxdt shape:", clar_dxdt.shape)
    # --- 8) Pack derivatives ---
    final_derivatives = np.concatenate([dydt_reactors.flatten(), clar_dxdt])
    # print("final_derivatives shape:", final_derivatives.shape)
    return final_derivatives
