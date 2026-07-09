import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import scipy.stats as stats
import plotly.express as px

# ------------------------------------------------------------------ #
# 1. The Core Math (Phase Focus)
# ------------------------------------------------------------------ #

def fit_sine(t, y, forced_freq=None):
    offset = np.mean(y)
    N = len(t)
    dt = (t[-1] - t[0]) / max(1, N - 1)
    
    if forced_freq is None:
        if dt <= 0:
            f_exact = 1.0 
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
        fit_phi = np.arctan2(b, a)
    except np.linalg.LinAlgError:
        fit_phi = 0.0
    
    return {"phase": fit_phi, "freq": f_exact}


def circular_mean(angles_rad):
    complex_vectors = np.exp(1j * np.array(angles_rad))
    return np.angle(np.mean(complex_vectors))

# ------------------------------------------------------------------ #
# 2. Chunking Logic
# ------------------------------------------------------------------ #

def process_file_chunks(csv_path, target_signal, target_cycles, skip_rows, unit):
    name = os.path.basename(csv_path)
    
    # Standardized Regex to match the Amplitude script
    match = re.search(r"stretch(\d+)(?:_twist(\d+))?_(sine|digital)_(\d+)hz", name, re.IGNORECASE)
    if not match: return None
    
    stretch, twist = int(match.group(1)), int(match.group(2) or 0)
    file_signal, nominal_freq = match.group(3).lower(), float(match.group(4))
    
    if file_signal != target_signal.lower() or twist == 360: return None
    
    df = pd.read_csv(csv_path).iloc[skip_rows:].reset_index(drop=True)
    if df.empty: return None 
    
    t, initial, distorted = df["timestamp"].values, df["voltage_initial_mv"].values, df["voltage_distorted_mv"].values
    
    total_time = t[-1] - t[0]
    if total_time <= 0: return None 
    
    window_time = min(total_time / 3, target_cycles / nominal_freq)
    chunk_starts = np.arange(t[0], t[-1] - window_time + 1e-6, window_time)
    if len(chunk_starts) == 0: chunk_starts = [t[0]]; window_time = total_time

    phase_shifts_rad = []
    
    for start in chunk_starts:
        mask = (t >= start) & (t < start + window_time)
        t_chunk = t[mask]
        i_chunk = initial[mask]
        d_chunk = distorted[mask]
        
        if len(t_chunk) < 10: continue
            
        fit_i = fit_sine(t_chunk, i_chunk)
        fit_d = fit_sine(t_chunk, d_chunk, forced_freq=fit_i["freq"])
        
        delta_phi = fit_d["phase"] - fit_i["phase"]
        delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi
        phase_shifts_rad.append(delta_phi)

    if not phase_shifts_rad: return None

    unwrapped_shifts = np.unwrap(phase_shifts_rad)
    
    if unit == "sec":
        shifts_array = unwrapped_shifts / (2 * np.pi * nominal_freq)
        col_name = "Time Shift (s)"
    elif unit == "rad":
        shifts_array = unwrapped_shifts
        col_name = "Phase Shift (rad)"
    else:
        shifts_array = np.degrees(unwrapped_shifts)
        col_name = "Phase Shift (deg)"
    
    avg_val = np.mean(shifts_array)
    
    return {
        "Stretch": stretch,
        "Twist": twist,
        "Frequency (Hz)": nominal_freq,
        col_name: avg_val,
        "raw_shifts": shifts_array 
    }

# ------------------------------------------------------------------ #
# 3. Matrix & Statistics Builder
# ------------------------------------------------------------------ #

def build_shift_matrix(input_dir, signal, target_cycles, skip_rows, unit, stats_enabled):
    files = glob.glob(os.path.join(input_dir, "*.csv"))
    data = [process_file_chunks(f, signal, target_cycles, skip_rows, unit) for f in files]
    
    data = [d for d in data if d is not None]
    df = pd.DataFrame(data)
    
    if df.empty: return None, None, None

    val_col = [col for col in df.columns if "Shift" in col][0]

    pval_matrix = None
    if stats_enabled:
        baseline_df = df[(df["Stretch"] == 0) & (df["Twist"] == 0)]
        print(f"Found {len(baseline_df)} baseline files for statistical comparison.")
        
        def calculate_p(row):
            baseline_rows = baseline_df[baseline_df["Freq"] == row["Frequency (Hz)"]]
            if not baseline_rows.empty:
                base_arr = baseline_rows.iloc[0]["raw_shifts"]
                if len(row["raw_shifts"]) > 1 and len(base_arr) > 1:
                    _, p = stats.ttest_ind(row["raw_shifts"], base_arr, equal_var=False)
                    return p
            return np.nan

        df["P-Value"] = df.apply(calculate_p, axis=1)
        pval_matrix = df.pivot_table(index=["Stretch", "Twist"], columns="Frequency (Hz)", values="P-Value")

    shift_matrix = df.pivot_table(index=["Stretch", "Twist"], columns="Frequency (Hz)", values=val_col)
    
    if stats_enabled and pval_matrix is not None:
        pval_matrix = pval_matrix.reindex(index=shift_matrix.index, columns=shift_matrix.columns)
        
    return shift_matrix, pval_matrix, val_col

def plot_matrix(matrix, pval_matrix, val_col, show, save_path, decimals=4):
    plot_df = matrix.copy()
    # Standardized labels to match Amplitude matrix ("Stretch: X%")
    plot_df.index = [f"Stretch: {s}%, Twist: {t}°" for s, t in matrix.index]
    plot_df.columns = [f"{int(f) if f.is_integer() else f}" for f in plot_df.columns]
    
    text_data = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix.columns)):
            val = matrix.iloc[i, j]
            if pd.isna(val): row.append(""); continue
            
            label = f"{val:.{decimals}f}"
            
            if pval_matrix is not None and i < len(pval_matrix) and j < len(pval_matrix.columns):
                p = pval_matrix.iloc[i, j]
                if pd.notna(p):
                    stars = "**" if p < 0.01 else "*" if p < 0.05 else ""
                    label += f"{stars}<br>p={p:.3f}"
            row.append(label)
        text_data.append(row)
    
    # Diverging color scale requires centered zmin/zmax
    max_abs_val = np.abs(plot_df.values).max()
    base_title = val_col.split(" (")[0]
    
    fig = px.imshow(
        plot_df,
        text_auto=False, 
        labels=dict(x="Frequency (Hz)", y="Sensor State", color=val_col),
        aspect="auto",
        color_continuous_scale="RdBu_r", 
        zmin=-max_abs_val,
        zmax=max_abs_val
    )
    
    # --- STANDARDIZED SIZES ---
    fig.update_traces(
        text=text_data, 
        texttemplate="%{text}",
        textfont=dict(size=22) 
    )

    fig.update_layout(
        title=dict(
            text=f"Sensor {base_title} Matrix",
            font=dict(size=30)
        ),
        xaxis_title="Frequency (Hz)",
        yaxis_title="Sensor State",
        autosize=True,
        margin=dict(l=150, r=50, t=100, b=100), 
        coloraxis_colorbar=dict(
            thickness=30,
            title_font=dict(size=22),
            tickfont=dict(size=20)
        ),
        font=dict(size=20) 
    )
    
    fig.update_xaxes(type='category', title_font=dict(size=22))
    fig.update_yaxes(title_font=dict(size=22))
    
    if save_path: fig.write_html(save_path)
    if show: fig.show()

# ------------------------------------------------------------------ #
# 4. CLI
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Process DAQ CSV files to build a Statistical Phase Shift Matrix.")
    parser.add_argument("input_dir", help="Path to DAQ CSV files.")
    parser.add_argument("--signal", choices=["sine", "digital"], default="sine")
    parser.add_argument("--unit", choices=["deg", "rad", "sec"], default="deg")
    parser.add_argument("--target-cycles", type=int, default=15)
    parser.add_argument("--skip-rows", type=int, default=10)
    parser.add_argument("--stats", action="store_true", help="Calculate and show p-values vs Baseline.")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save-plot", nargs="?", const="stat_shift_matrix.html", default=None)
    parser.add_argument("--decimals", type=int, default=4, help="Number of decimal places to show.")
    
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Directory '{args.input_dir}' does not exist.")
        return

    shift_matrix, pval_matrix, val_col = build_shift_matrix(
        args.input_dir, args.signal, args.target_cycles, args.skip_rows, args.unit, args.stats
    )

    if shift_matrix is not None:
        print(f"\n--- FINAL MATRIX ({val_col}) ---")
        print(shift_matrix.round(args.decimals))
        
        if args.stats:
            print(f"\n--- P-VALUES (Welch's T-Test vs Stretch 0, Twist 0) ---")
            print(pval_matrix.fillna("N/A").round(4))
        
        if args.plot or args.save_plot:
            plot_matrix(shift_matrix, pval_matrix, val_col, args.plot, args.save_plot, args.decimals)
    else:
        print("\nCould not build matrix. Check if the directory contains matching CSV files.")

if __name__ == "__main__":
    main()