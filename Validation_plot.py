# Validation_plot.py
# No LaTeX, no pandas. Manual short x-axis labels.

import os, csv
import numpy as np
import matplotlib.pyplot as plt

CSV_NAME = "bsm1_validation_summary.csv"
PNG_OUT  = "bsm1_validation_latex.png"
PDF_OUT  = "bsm1_validation_latex.pdf"

# Where your CSV lives if not in CWD
FALLBACK_DIR = "/lustre/isaac24/scratch/mshatara/Codes/BSM1/results_takacs/Takacas_validation"

# <<< Edit these to your taste (must match number/order of rows in CSV) >>>
SHORT_TICK_LABELS = [
    "COD",        # CODt_avg
    "TN",         # Ntot_avg
    r"NH4-N",     # SNH_avg
    "TSS",        # TSS_avg
    "BOD5",       # BOD5_avg
    r"NH4-N (95)",# SNH95
    "TN (95)",    # Ntot95
    "TSS (95)",   # TSSe95
]

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


if __name__ == "__main__":
    main()
