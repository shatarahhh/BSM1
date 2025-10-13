# plot_influent_fit_save.py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend for terminals/servers
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, UnivariateSpline
from pathlib import Path
import argparse

def load_influent(file_path: Path) -> pd.DataFrame:
    cols = ['time','S_I','S_S','X_I','X_S','X_H','X_A','X_P',
            'S_O','S_NO','S_NH','S_ND','X_ND','S_ALK','Q_in']
    df = pd.read_csv(file_path, sep="\t", header=None, names=cols, skiprows=1)
    return df.sort_values('time').reset_index(drop=True)

def make_model(x, y, kind="linear", smooth=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if smooth is None:
        f = interp1d(
            x, y, kind=kind,
            bounds_error=False,
            fill_value=(y[0], y[-1]) # type: ignore
        )
        return lambda t: np.asarray(f(t), dtype=float), f"{kind}"
    else:
        spl = UnivariateSpline(x, y, s=float(smooth))
        return lambda t: np.asarray(spl(t), dtype=float), f"spline_s{smooth}"

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred)**2) # type: ignore
    ss_tot = np.sum((y_true - np.mean(y_true))**2) # type: ignore
    return np.nan if ss_tot == 0 else (1.0 - ss_res/ss_tot)

def main():
    ap = argparse.ArgumentParser(description="Plot discrete influent samples vs continuous model and save images.")
    ap.add_argument("--file", type=Path, required=True, help="Path to BSM1 influent file (e.g., influent/Inf_dry_2006.txt)")
    ap.add_argument("--component", type=str, default="Q_in",
                    help="One of 'Q_in' or a component column: S_I,S_S,X_I,X_S,X_H,X_A,X_P,S_O,S_NO,S_NH,S_ND,X_ND,S_ALK")
    ap.add_argument("--kind", type=str, default="linear",
                    help="interp1d kind if smooth is None: linear | previous | nearest | cubic ...")
    ap.add_argument("--smooth", type=float, default=None,
                    help="Smoothing spline parameter s; if set, uses UnivariateSpline instead of exact interpolation")
    ap.add_argument("--n_fine", type=int, default=2000, help="Number of points for the continuous curve")
    ap.add_argument("--outdir", type=Path, default=Path("results_influent_fit"), help="Output directory for images")
    ap.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    df = load_influent(args.file)

    if args.component == "Q_in":
        y = df["Q_in"].values
        y_label = "Q_in (m³/d)"
    else:
        if args.component not in df.columns:
            raise SystemExit(f"Component {args.component} not found in file.")
        y = df[args.component].values
        y_label = f"{args.component} (g/m³)" if args.component != "S_ALK" else "S_ALK (mol/m³)"

    x = df["time"].values

    f, model_tag = make_model(x, y, kind=args.kind, smooth=args.smooth)

    # R² at the original sample times
    y_hat_samples = f(x)
    r2 = r2_score(y, y_hat_samples)
    print(f"R² at sample points for {args.component} using {model_tag}: {r2:.6f}")

    # Dense curve
    x_fine = np.linspace(x.min(), x.max(), args.n_fine) # type: ignore
    y_fine = f(x_fine)

    # Filenames
    base = f"{args.component}_{model_tag}".replace(" ", "")
    fit_png = args.outdir / f"{base}_fit.png"
    resid_png = args.outdir / f"{base}_residuals.png"

    # Fit plot
    plt.figure(figsize=(12, 6))
    plt.scatter(x, y, s=18, alpha=0.8, label="Discrete samples") # type: ignore
    plt.plot(x_fine, y_fine, lw=2.0, label=f"Continuous model ({model_tag})")
    plt.title(f"{args.component}: samples vs continuous model | R²={r2:.5f}")
    plt.xlabel("Time (days)")
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fit_png, dpi=args.dpi)
    plt.close()

    # Residuals plot
    residuals = y - y_hat_samples # type: ignore
    plt.figure(figsize=(12, 3.5))
    plt.axhline(0, lw=1)
    plt.scatter(x, residuals, s=14, alpha=0.8) # type: ignore
    plt.title("Residuals (observed − model) at sample times")
    plt.xlabel("Time (days)")
    plt.ylabel("Residual")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(resid_png, dpi=args.dpi)
    plt.close()

    print(f"Saved:\n  {fit_png}\n  {resid_png}")

if __name__ == "__main__":
    main()
