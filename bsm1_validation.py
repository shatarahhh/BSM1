# bsm1_validation.py File
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

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
    dt15  = 1.0 / 96.0                       # 15 min in days
    t_kpi = np.arange(t_start, t_end, dt15)  # [7.0, 14.0)

    # ----- 2) Sample effluent flow on the KPI grid: Qe = Q0 - Qw (BSM1) -----
    Q_w    = float(clarifier_params['Q_w'])
    Qe_kpi = np.array([float(get_influent_data(tt)[0]) for tt in t_kpi]) - Q_w
    # (optional) hardening against negatives:
    # Qe_kpi = np.maximum(Qe_kpi, 0.0)

    # ----- 3) Sample effluent concentrations on the KPI grid (ZOH / left-endpoint) -----
    # For each 15‑min timestamp, take the last available solver sample at or before it.
    # eff_kpi = np.column_stack([np.interp(t_kpi, t_span, effluent_ts[:, j]) for j in range(13)])
    eff_kpi = effluent_ts[np.clip(np.searchsorted(t_span, t_kpi, side='right') - 1, 0, len(t_span) - 1), :]     # shape (len(t_kpi), 13)

    # Column index reminder (ASM1 order)
    # 0:S_I, 1:S_S, 2:X_I, 3:X_S, 4:X_H, 5:X_A, 6:X_P, 7:S_O, 8:S_NO, 9:S_NH, 10:S_ND, 11:X_ND, 12:S_ALK

    # ----- 4) Derived effluent series per BSM1 definitions -----
    part_idx = [2, 3, 4, 5, 6]
    COD_idx  = [0, 1, 2, 3, 4, 5, 6]
    TSS_e    = 0.75 * eff_kpi[:, part_idx].sum(axis=1)
    CODt_e   =        eff_kpi[:, COD_idx].sum(axis=1)

    f_P    = float(stoich_params['f_P'])
    BOD5_e = 0.25 * (eff_kpi[:, 1] + eff_kpi[:, 3] + (1.0 - f_P) * (eff_kpi[:, 4] + eff_kpi[:, 5]))

    i_XB   = float(stoich_params['i_XB'])
    i_XP   = float(stoich_params['i_XP'])
    Nkj_e  = (eff_kpi[:, 9] + eff_kpi[:, 10] + eff_kpi[:, 11]
              + i_XB * (eff_kpi[:, 4] + eff_kpi[:, 5]) + i_XP * eff_kpi[:, 6])
    Ntot_e = eff_kpi[:, 8] + Nkj_e

    # ----- 5) Flow-weighted averages (ZOH over [t_start, t_end)) -----
    w = Qe_kpi * dt15

    def flow_avg(series_1d):
        return float((w * series_1d).sum() / w.sum())

    # ----- 6) KPIs -----
    results = {
        'CODt_avg': flow_avg(CODt_e),
        'Ntot_avg': flow_avg(Ntot_e),
        'SNH_avg' : flow_avg(eff_kpi[:, 9]),
        'TSS_avg' : flow_avg(TSS_e),
        'BOD5_avg': flow_avg(BOD5_e),
        # 95th percentiles on 15-min samples (unweighted) within the same window
        'SNH95'   : float(np.percentile(eff_kpi[:, 9], 95)),
        'Ntot95'  : float(np.percentile(Ntot_e,       95)),
        'TSS95'   : float(np.percentile(TSS_e,        95)),
    }
    return results

def make_bsm1_validation_plot(kpis: dict,
                              TSSe95: float,
                              out_png: str = "results/bsm1_validation.png",
                              out_csv: str = "results/bsm1_validation_summary.csv") -> None:
    """
    Create a publication-quality bar chart comparing your KPIs to the BSM1 benchmark
    and save a CSV with absolute and percent differences.

    The figure follows BSM1 conventions: days 7–14, 15-min rectangular integration.
    """
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    # Benchmark (original model) values you provided
    REF_BSM1 = {
        "CODt_avg": 48.30,     # g COD m^-3
        "Ntot_avg": 15.58,     # g N   m^-3
        "SNH_avg" : 4.794,     # g N   m^-3
        "TSS_avg" : 12.99,     # g SS  m^-3
        "BOD5_avg": 2.775,     # g     m^-3
        "SNH95"   : 8.9175,    # g N   m^-3
        "Ntot95"  : 18.535,    # g N   m^-3
        "TSSe95"  : 15.8,      # g SS  m^-3
    }

    # Merge your KPIs and your TSSe95 into one dict
    model = dict(kpis)
    model["TSSe95"] = float(TSSe95)

    # Metric list (key, pretty label, unit)
    items = [
        ("CODt_avg", r"Total COD (avg)",      r"g COD m$^{-3}$"),
        ("Ntot_avg", r"Total N (avg)",        r"g N m$^{-3}$"),
        ("SNH_avg",  r"NH$_4$-N (avg)",       r"g N m$^{-3}$"),
        ("TSS_avg",  r"TSS (avg)",            r"g SS m$^{-3}$"),
        ("BOD5_avg", r"BOD$_5$ (avg)",        r"g m$^{-3}$"),
        ("SNH95",    r"NH$_4$-N (95th)",      r"g N m$^{-3}$"),
        ("Ntot95",   r"Total N (95th)",       r"g N m$^{-3}$"),
        ("TSSe95",   r"TSS (95th)",           r"g SS m$^{-3}$"),
    ]

    labels = [f"{name}\n[{unit}]" for _, name, unit in items]
    ref_vals = np.array([REF_BSM1[k] for k, _, _ in items], dtype=float)
    our_vals = np.array([float(model[k]) for k, _, _ in items], dtype=float)
    diffs_abs = our_vals - ref_vals # type: ignore
    diffs_pct = 100.0 * diffs_abs / ref_vals

    # Save CSV summary
    df = pd.DataFrame({
        "metric_key": [k for k, _, _ in items],
        "label":      [name for _, name, _ in items],
        "unit":       [unit for _, _, unit in items],
        "benchmark":  ref_vals,
        "this_work":  our_vals,
        "abs_diff":   diffs_abs,
        "pct_diff":   diffs_pct,
    })
    df.to_csv(out_csv, index=False)

    # Plot: grouped bars + % difference labels (no custom colors, single axes)
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, ref_vals, width, label="Benchmark (BSM1)") # type: ignore
    ax.bar(x + width/2, our_vals, width, label="This work") # type: ignore

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Value in original units")
    ax.set_title("BSM1 KPI validation (days 7–14; 15-min rectangular integration)")
    ax.grid(True, axis="y")
    ax.legend()

    # Annotate % difference above "This work" bars
    for xi, yi, di in zip(x + width/2, our_vals, diffs_pct):
        ax.text(xi, yi, f"{di:+.2f}%", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Also print a concise console summary
    print("\n[Validation vs BSM1]")
    for (k, name, _), r, o, da, dp in zip(items, ref_vals, our_vals, diffs_abs, diffs_pct):
        print(f"{name:18s}: ours={o:.3f} | ref={r:.3f} | Δ={da:+.3f} ({dp:+.2f}%)")
    print(f"Plot saved → {out_png}")
    print(f"CSV  saved → {out_csv}")
