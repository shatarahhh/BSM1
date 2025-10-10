# bsm1_validation.py File
import numpy as np

def compute_bsm1_kpis(t_span,
                      effluent_ts,            # shape (T, 13) from your simulation grid
                      get_influent_data,      # function(t) -> (Q0, Zin)
                      clarifier_params,       # provides Q_w
                      stoich_params,          # provides f_P, i_XB, i_XP
                      t_start=7.0, t_end=14.0 # evaluation window [7,14) days
                      ):
    """
    Compute the official BSM1 KPIs on the mandated 15-min grid
    using rectangular (zero-order hold) integration over days 7–14.

    Returns a dict with flow-weighted means and 95th percentiles.
    """

    # ----- 1) Build the 15-min KPI grid (left endpoints for rectangular/ZOH) -----
    dt15  = 1.0 / 96.0                           # 15 min in days
    t_kpi = np.arange(0.0, 14.0, dt15)           # 0, 0.15625, ..., 13.84375

    # ----- 2) Sample effluent flow on the KPI grid: Qe = Q0 - Qw (BSM1) -----
    # The influent function returns flow at any t; Q_w is constant in BSM1.
    Q_w    = float(clarifier_params['Q_w'])
    Qe_kpi = np.array([float(get_influent_data(tt)[0]) for tt in t_kpi]) - Q_w

    # ----- 3) Sample effluent concentrations on the KPI grid (linear to samples, then ZOH for integration) -----
    # effluent_ts is on your solver grid t_span; we interpolate each of the 13 components to t_kpi.
    eff_kpi = np.column_stack([np.interp(t_kpi, t_span, effluent_ts[:, j]) for j in range(13)])

    # Column index reminder (ASM1 order)
    # 0:S_I, 1:S_S, 2:X_I, 3:X_S, 4:X_H, 5:X_A, 6:X_P, 7:S_O, 8:S_NO, 9:S_NH, 10:S_ND, 11:X_ND, 12:S_ALK

    # ----- 4) Derived effluent series per BSM1 definitions -----
    # TSS_e = 0.75 * sum of particulate COD (X_I, X_S, X_H, X_A, X_P)    [BSM1 Section 2.3.3]
    # CODt_e = total COD = S_I + S_S + X_I + X_S + X_H + X_A + X_P       [BSM1 Section 6]
    part_idx = [2, 3, 4, 5, 6]
    COD_idx  = [0, 1, 2, 3, 4, 5, 6]
    TSS_e    = 0.75 * eff_kpi[:, part_idx].sum(axis=1)
    CODt_e   =        eff_kpi[:, COD_idx].sum(axis=1)

    # BOD5_e = 0.25 * (S_S + X_S + (1 - f_P) * (X_H + X_A))              [BSM1 Section 6]
    f_P      = float(stoich_params['f_P'])
    BOD5_e   = 0.25 * (eff_kpi[:, 1] + eff_kpi[:, 3] + (1.0 - f_P) * (eff_kpi[:, 4] + eff_kpi[:, 5]))

    # N_kj = S_NH + S_ND + X_ND + i_XB*(X_H+X_A) + i_XP*X_P; N_tot = S_NO + N_kj   [BSM1 Section 6]
    i_XB     = float(stoich_params['i_XB'])
    i_XP     = float(stoich_params['i_XP'])
    Nkj_e    = (eff_kpi[:, 9] + eff_kpi[:, 10] + eff_kpi[:, 11]
                + i_XB * (eff_kpi[:, 4] + eff_kpi[:, 5]) + i_XP * eff_kpi[:, 6])
    Ntot_e   = eff_kpi[:, 8] + Nkj_e

    # ----- 5) Select the evaluation window [7, 14) on the KPI grid -----
    mask = (t_kpi >= t_start) & (t_kpi < t_end)
    w    = Qe_kpi[mask] * dt15   # rectangular (ZOH) weights = Qe * Δt   [BSM1 Section 4]

    # Helper for flow-weighted average on the window
    def flow_avg(series_1d):
        s = series_1d[mask]
        return float((w * s).sum() / w.sum())

    # ----- 6) Flow-weighted means (BSM1) -----
    results = {
        'CODt_avg': flow_avg(CODt_e),
        'Ntot_avg': flow_avg(Ntot_e),
        'SNH_avg' : flow_avg(eff_kpi[:, 9]),
        'TSS_avg' : flow_avg(TSS_e),
        'BOD5_avg': flow_avg(BOD5_e),
        # 95th percentiles on 15-min samples (unweighted) over the same window   [BSM1 Section 6]
        'SNH95'   : float(np.percentile(eff_kpi[mask, 9], 95)),
        'Ntot95'  : float(np.percentile(Ntot_e[mask],     95)),
        'TSS95'   : float(np.percentile(TSS_e[mask],      95)),
    }
    return results
