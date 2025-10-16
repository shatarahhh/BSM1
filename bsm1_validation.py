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


##############################################################################


# separate code for plotting validation in a better way


def find_csv():
    here = os.getcwd()
    p1 = os.path.join(here, CSV_NAME)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(FALLBACK_DIR, CSV_NAME)
    if os.path.exists(p2):
        return p2
    raise FileNotFoundError(f"Could not find {CSV_NAME} in {here} or {FALLBACK_DIR}")

def load_summary(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        raise ValueError("CSV has no rows.")
    ref_vals = np.array([float(r["benchmark"]) for r in rows], dtype=float)
    our_vals = np.array([float(r["this_work"]) for r in rows], dtype=float)
    pct_diff = np.array([float(r["pct_diff"]) for r in rows], dtype=float)
    return rows, ref_vals, our_vals, pct_diff

def set_style():
    plt.rcParams.clear()
    # plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.family"] = "Nimbus Roman"
    plt.rcParams["text.usetex"] = False
    plt.rcParams.update({"font.size": 9})
    plt.rcParams["lines.linewidth"] = 1

def make_plot(tick_labels, ref_vals, our_vals, pct_diff, out_dir):
    x = np.arange(len(tick_labels))
    width = 0.37

    fig, ax = plt.subplots(figsize=(6.5, 3.2))

    ax.bar(x - width/2, ref_vals, width, label="Benchmark (BSM1)")
    ax.bar(x + width/2, our_vals, width, label="Our BSM1 simulation")

    ax.set_ylabel("Value")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=0, ha="center")
    fig.subplots_adjust(bottom=0.2)

    ax.grid(True, axis="y", linestyle="--", alpha=0.35, linewidth=0.8)
    ax.legend(frameon=False, loc="upper right")

    ymax = float(max(ref_vals.max(), our_vals.max()))
    ax.set_ylim(0, ymax * 1.20)

    for xi, yi, di in zip(x + width/2, our_vals, pct_diff):
        ax.text(xi, yi, f"{di:+.2f}%", ha="center", va="bottom", fontsize=9)

    png_path = os.path.join(out_dir, PNG_OUT)
    pdf_path = os.path.join(out_dir, PDF_OUT)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {png_path}")
    print(f"Saved → {pdf_path}")

def main():
    csv_path = find_csv()
    out_dir = os.path.dirname(csv_path)
    set_style()
    rows, ref_vals, our_vals, pct_diff = load_summary(csv_path)

    # Safety: ensure manual labels length matches CSV rows
    if len(SHORT_TICK_LABELS) != len(rows):
        raise ValueError(f"SHORT_TICK_LABELS has {len(SHORT_TICK_LABELS)} items, "
                         f"but CSV has {len(rows)} rows. Please adjust the list.")

    # NEW: append units from CSV under each short label (second line)
    units = [r["unit"] for r in rows]

        # Reorder so BOD5 (row 4) is second: [COD, BOD5, TN, NH4-N, TSS, NH4(95), TN(95), TSS(95)]
    ORDER = [0, 4, 1, 2, 3, 5, 6, 7]

    # Reorder data + units
    ref_vals = ref_vals[ORDER]
    our_vals = our_vals[ORDER]
    pct_diff = pct_diff[ORDER]
    units = [units[i] for i in ORDER]

    # Reorder your short labels to match
    short_labels = [SHORT_TICK_LABELS[i] for i in ORDER]

    # Build final labels (short + units)
    final_labels = [f"{short}\n[{unit}]" for short, unit in zip(short_labels, units)]

    # final_labels = [f"{short}\n[{unit}]" for short, unit in zip(SHORT_TICK_LABELS, units)]

    make_plot(final_labels, ref_vals, our_vals, pct_diff, out_dir)


# if __name__ == "__main__":

#     CSV_NAME = "bsm1_validation_summary.csv"
#     PNG_OUT  = "bsm1_validation_latex.png"
#     PDF_OUT  = "bsm1_validation_latex.pdf"

#     # Where your CSV lives if not in CWD
#     FALLBACK_DIR = "/lustre/isaac24/scratch/mshatara/Codes/BSM1/results_takacs/Takacas_validation"

#     # <<< Edit these to your taste (must match number/order of rows in CSV) >>>
#     SHORT_TICK_LABELS = [
#         "COD",        # CODt_avg
#         "TN",         # Ntot_avg
#         r"NH4-N",     # SNH_avg
#         "TSS",        # TSS_avg
#         "BOD5",       # BOD5_avg
#         r"NH4-N (95)",# SNH95
#         "TN (95)",    # Ntot95
#         "TSS (95)",   # TSSe95
#     ]
#     main()
