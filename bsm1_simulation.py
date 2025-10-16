# bsm1_simulation.py File

import numpy as np
from ASM1_Processes import calculate_process_rates
from clarifier_model import takacs_clarifier_model, takacs_clarifier_soluble_odes  
from NN import run_prediction_pipeline

# indices: [2..6] are particulate COD; 11 is X_ND (particulate N)
particulate_cod_idx  = [2, 3, 4, 5, 6] # X_I, X_S, X_H, X_A, X_P
all_particulate_idx  = [2, 3, 4, 5, 6, 11] # add X_ND
soluble_idx = [0, 1, 7, 8, 9, 10, 12]  # S_I, S_S, S_O, S_NO, S_NH, S_ND, S_ALK

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

# needs double checking
# def NN_project_to_mass_balance(Xe_hat, Xu_hat, Xf, Q_to_clarifier, Qe, Qu,
#                             w_e=1.0, w_u=0.2, bounds=(0.0, 12000.0)):
#     Xe_hat = float(Xe_hat); Xu_hat = float(Xu_hat)
#     S = Q_to_clarifier * Xf
#     denom = (Qe**2)/w_e + (Qu**2)/w_u
#     if denom <= 0.0:
#         Xe, Xu = Xe_hat, Xu_hat
#     else:
#         lam = (S - Qe*Xe_hat - Qu*Xu_hat) / denom
#         Xe = Xe_hat + lam * (Qe / w_e)
#         Xu = Xu_hat + lam * (Qu / w_u)
#     lo, hi = bounds
#     Xe = float(np.clip(Xe, lo, hi)); Xu = float(np.clip(Xu, lo, hi))
#     if Xu < Xe:  # optional, typical physical guardrail
#         mid = 0.5*(Xe+Xu); Xe, Xu = min(Xe, mid), max(Xu, mid)
#     return Xe, Xu

def bsm1_plant_model(y, t, influent_data, stoich_params, Kin_params, clarifier_params, settling_params,
                     nn_model, nn_u_scaler, nn_y_scaler, nn_s_scaler, nn_y_indices, stab):
    """
    BSM1 plant dynamics (5 reactors + 10-layer clarifier).
    y has 145 states: 5*13 (reactors) + 10 (clarifier MLSS) + 10*7 (clarifier solubles).
    """

    # --- 1) Unpack state vector ---
    y_reactors   = y[0:65]
    X_reactors   = y_reactors.reshape(5, 13)     # 5 tanks x 13 ASM1 states
    y_clarifier  = y[65:75]                      # 10-layer MLSS profile (top->bottom or your chosen convention)

    # 10 layers × 7 solubles, flat after the 10 MLSS states
    y_solubles   = y[75: 145]   # flat soluble layers
    Z_layers     = y_solubles.reshape(10, 7)  # (10,7), top->bottom 

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

    # Clarifier feed flow (BSM1): Q_to_clarifier = Q5 - Qa = influent_flow + Qr
    Q_RAS = clarifier_params['Q_RAS']  # return sludge (underflow to Tank 1)
    Q_to_clarifier = influent_flow + Q_RAS  # do NOT add Q_IR
    Q_w   = clarifier_params['Q_w']          #  wastage
    Q_IR  = 55338                      # internal recycle (Tank5 -> Tank1)
    Qu    = Q_RAS + Q_w
    Qe    = Q_to_clarifier - Qu  # = influent_flow - Q_w

    # Reactor through-flow is constant in all tanks:
    # Q_reac = influent_flow + Qa + Qr
    outflow_rate = influent_flow + Q_IR + Q_RAS

    # --- 4) Clarifier feed & MLSS  ---
    # MLSS entering clarifier = 0.75 * (sum of particulate COD in Tank 5)
    total_particulate_cod_t5 = np.sum(X_reactors[4, particulate_cod_idx])
    MLSS_in_clarifier = 0.75 * total_particulate_cod_t5

    # --- 5) Clarifier model: Takács (dynamic) or NN steady-state (algebraic) ---
    if clarifier_params.get('mode', 'takacs') == 'nn_ss' and stab is False:
        Q_unit_scale = 1/24 # m3/d (bsm1) to m3/h (NN)
        MLSS_unit_scale = 1/1000 # mg/L or g/m3 (bsm1) to g/L or kg/m3 (NN)
        MLSS_in_clarifier_scaled, Q_to_clarifier_scaled, Qe_scaled, Qu_scaled = MLSS_in_clarifier * MLSS_unit_scale, Q_to_clarifier * Q_unit_scale, Qe * Q_unit_scale, Qu * Q_unit_scale
        Xe_hat_scaled, Xu_hat_scaled = run_prediction_pipeline(Q_to_clarifier_scaled, Qu_scaled, MLSS_in_clarifier_scaled, settling_params, 
            nn_model, nn_u_scaler, nn_y_scaler, nn_s_scaler, nn_y_indices, plot=False)
        print(f"NN (g/L): Xe_hat_scaled={Xe_hat_scaled:.7f}, Xu_hat_scaled={Xu_hat_scaled:.7f}")
        # Optional: exact steady solids balance (recommended)
        Xe_scaled, Xu_scaled = float(max(0.0, Xe_hat_scaled)), float(max(0.0, Xu_hat_scaled))
        # if clarifier_params.get('project_balance', True):
        #     Xe_scaled, Xu_scaled = NN_project_to_mass_balance(Xe_hat_scaled, Xu_hat_scaled, MLSS_in_clarifier_scaled, Q_to_clarifier_scaled, Qe_scaled, Qu_scaled)
        Xe, Xu = Xe_scaled / MLSS_unit_scale, Xu_scaled / MLSS_unit_scale
        # print(f"NN (g/L): Xe={Xe:.3f}, Xu={Xu:.1f}")


        # No MLSS dynamics under instantaneous steady assumption
        clar_dxdt = np.zeros(clarifier_params['N_layers'], dtype=float)

        # Keep soluble advection ODEs (unchanged hydraulics)
        Z_in_feed = X_reactors[4, soluble_idx]
        dZdt_flat, Zu_vec, Ze_vec = takacs_clarifier_soluble_odes(Z_layers, Z_in_feed, Q_to_clarifier, clarifier_params)

        # Build RAS from Xu (particulates scaled), solubles from bottom layer
        RAS_concs_full = map_particulates_by_tss(X_reactors[4, :], Xu, MLSS_in_clarifier)
        RAS_concs_full[soluble_idx] = Zu_vec

        clarifier_params['last_Xe'] = float(Xe) # I added this because in the case of the NN, the clarifier doesnt use derivatives so i need to send the solution at ESS and Xras instead.
        clarifier_params['last_Xu'] = float(Xu)


    else:

        clar_dxdt, Xu, Xe = takacs_clarifier_model(
            y_clarifier,
            MLSS_in_clarifier,
            Q_to_clarifier,
            clarifier_params,
            settling_params
        )

        # --- 5b) Run soluble-layer advection ODEs ---
        Z_in_feed = X_reactors[4, soluble_idx]  # Tank 5 solubles at the settler influent
        dZdt_flat, Zu_vec, Ze_vec = takacs_clarifier_soluble_odes(
            Z_layers, Z_in_feed, Q_to_clarifier, clarifier_params
        )

        # --- 6) Build RAS composition by MLSS ratio (particulates scaled, solubles copied) ---
        RAS_concs_full = map_particulates_by_tss(X_reactors[4, :], Xu, MLSS_in_clarifier)

        # underflow solubles = bottom-layer solubles (Zu)
        RAS_concs_full[soluble_idx] = Zu_vec

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
    final_derivatives = np.concatenate([dydt_reactors.flatten(), clar_dxdt, dZdt_flat])
    # print("final_derivatives shape:", final_derivatives.shape)
    return final_derivatives