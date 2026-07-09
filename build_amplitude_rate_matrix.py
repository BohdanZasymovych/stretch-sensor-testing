import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import scipy.stats as stats
import plotly.express as px


def fit_sine(t, y, forced_freq=None):
    offset = np.mean(y)
    N = len(t)
    dt = (t[-1] - t[0]) / max(1, N - 1)
    
    if forced_freq is None:
        if dt <= 0: f_exact = 1.0 
        else:
            yf = np.fft.fft(y - offset)
            xf = np.fft.fftfreq(N, dt)
            f_rough = xf[np.argmax(np.abs(yf[1:N//2])) + 1]
            if not np.isfinite(f_rough): f_rough = 1.0
            
            def negative_amplitude(f):
                w = 2 * np.pi * f
                X = np.column_stack([np.sin(w * t), np.cos(w * t), np.ones_like(t)])
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                    return -np.sqrt(coeffs[0]**2 + coeffs[1]**2)
                except np.linalg.LinAlgError:
                    return 0.0

            res = minimize_scalar(negative_amplitude, bounds=(f_rough - 2.0, f_rough + 2.0), method='bounded')
            f_exact = res.x
    else:
        f_exact = forced_freq

    w_exact = 2 * np.pi * f_exact
    X = np.column_stack([np.sin(w_exact * t), np.cos(w_exact * t), np.ones_like(t)])
    
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        a, b, C = coeffs
        fit_amp = np.sqrt(a**2 + b**2)
    except np.linalg.LinAlgError:
        fit_amp = 0.0
    
    return {"freq": f_exact, "amplitude": fit_amp}


def process_file_chunks(csv_path, target_signal, target_cycles, skip_rows):
    name = os.path.basename(csv_path)
    match = re.search(r"stretch(\d+)(?:_twist(\d+))?_(sine|digital)_(\d+)hz", name, re.IGNORECASE)
    if not match: return None
    
    stretch, twist = int(match.group(1)), int(match.group(2) or 0)
    file_signal, nominal_freq = match.group(3).lower(), float(match.group(4))
    
    if file_signal != target_signal.lower() or twist == 360: return None
    
    df = pd.read_csv(csv_path).iloc[skip_rows:].reset_index(drop=True)
    t, initial, distorted = df["timestamp"].values, df["voltage_initial_mv"].values, df["voltage_distorted_mv"].values
    
    total_time = t[-1] - t[0]
    window_time = min(total_time / 3, target_cycles / nominal_freq)
    chunk_starts = np.arange(t[0], t[-1] - window_time + 1e-6, window_time)
    if len(chunk_starts) == 0: chunk_starts = [t[0]]; window_time = total_time

    ratios = []
    for start in chunk_starts:
        mask = (t >= start) & (t < start + window_time)
        fit_i = fit_sine(t[mask], initial[mask])
        fit_d = fit_sine(t[mask], distorted[mask], forced_freq=fit_i["freq"])
        if fit_i["amplitude"] > 0:
            ratios.append(fit_d["amplitude"] / fit_i["amplitude"])

    return {"Stretch": stretch, "Twist": twist, "Freq": nominal_freq, "raw_ratios": np.array(ratios)} if ratios else None


def build_matrix(input_dir, signal, target_cycles, skip_rows, subtract, percent, stats_enabled):
    files = glob.glob(os.path.join(input_dir, "*.csv"))
    data = [process_file_chunks(f, signal, target_cycles, skip_rows) for f in files]
    df = pd.DataFrame([d for d in data if d])
    
    if df.empty: return None, None, None

    df["Value"] = df["raw_ratios"].apply(np.mean)
    if subtract: df["Value"] = 1 - df["Value"]
    if percent: df["Value"] = df["Value"] * 100
    
    metric_label = "Loss" if subtract else "Ratio"
    metric_name = f"{metric_label}{'%' if percent else ''}"

    pval_matrix = None
    if stats_enabled:
        baseline_df = df[(df["Stretch"] == 0) & (df["Twist"] == 0)]
        print(f"Found {len(baseline_df)} baseline files for statistical comparison.")
        
        def calculate_p(row):
            baseline_rows = baseline_df[baseline_df["Freq"] == row["Freq"]]
            if not baseline_rows.empty:
                base_arr = baseline_rows.iloc[0]["raw_ratios"]
                if len(row["raw_ratios"]) > 1 and len(base_arr) > 1:
                    _, p = stats.ttest_ind(row["raw_ratios"], base_arr, equal_var=False)
                    return p
            return np.nan

        df["P-Value"] = df.apply(calculate_p, axis=1)
        pval_matrix = df.pivot_table(index=["Stretch", "Twist"], columns="Freq", values="P-Value")

    matrix = df.pivot_table(index=["Stretch", "Twist"], columns="Freq", values="Value")
    return matrix, pval_matrix, metric_name

def plot_matrix(matrix, pval_matrix, metric_name, show, save_path):
    plot_df = matrix.copy()
    plot_df.index = [f"Stretch: {s}, Twist: {t}°" for s, t in matrix.index]
    plot_df.columns = [f"{int(f) if f.is_integer() else f}" for f in plot_df.columns]
    
    text_data = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix.columns)):
            val = matrix.iloc[i, j]
            if pd.isna(val): row.append(""); continue
            label = f"{val:.4f}"
            if pval_matrix is not None and i < len(pval_matrix) and j < len(pval_matrix.columns):
                p = pval_matrix.iloc[i, j]
                if pd.notna(p):
                    stars = "**" if p < 0.01 else "*" if p < 0.05 else ""
                    label += f"{stars}<br>p={p:.3f}"
            row.append(label)
        text_data.append(row)
    
    fig = px.imshow(
        plot_df, 
        text_auto=False, 
        color_continuous_scale="Viridis", 
        labels={"color": metric_name},
        aspect="auto"
    )
        
    fig.update_traces(text=text_data, texttemplate="%{text}")

    fig.update_layout(
        title=f"Amplitude {metric_name} Matrix", 
        xaxis_title="Frequency (Hz)", 
        yaxis_title="Sensor State",
        autosize=True,
        margin=dict(l=150, r=50, t=80, b=80), 
        coloraxis_colorbar=dict(thickness=30)
    )
    
    fig.update_xaxes(type='category')
    
    if save_path: fig.write_html(save_path)
    if show: fig.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("--signal", choices=["sine", "digital"], default="sine")
    parser.add_argument("--percent", action="store_true")
    parser.add_argument("--subtract", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save-plot", nargs="?", const="matrix.html")
    args = parser.parse_args()

    matrix, pval_matrix, metric_name = build_matrix(args.input_dir, args.signal, 15, 10, args.subtract, args.percent, args.stats)
    
    if matrix is not None:
        print(f"\n--- {metric_name} Matrix ---"); print(matrix.round(4))
        plot_matrix(matrix, pval_matrix if args.stats else None, metric_name, args.plot, args.save_plot)


if __name__ == "__main__":
    main()