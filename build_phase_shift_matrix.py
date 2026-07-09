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
    """Fits a sine wave using linear least squares."""
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


def process_file_chunks(csv_path, target_cycles=15, skip_rows=10, unit="deg"):
    name = os.path.basename(csv_path)
    match = re.search(r"stretch(\d+)(?:_twist(\d+))?_sine_(\d+)hz", name, re.IGNORECASE)
    if not match: return None
    
    stretch = int(match.group(1))
    twist = int(match.group(2)) if match.group(2) else 0
    nominal_freq = float(match.group(3))
    
    if twist == 360: return None
    
    df = pd.read_csv(csv_path).iloc[skip_rows:].reset_index(drop=True)
    t = df["timestamp"].values
    initial = df["voltage_initial_mv"].values
    distorted = df["voltage_distorted_mv"].values
    
    total_time = t[-1] - t[0]
    window_time = target_cycles / nominal_freq
    
    min_chunks = 3
    if window_time > (total_time / min_chunks):
        window_time = total_time / min_chunks
        
    chunk_starts = np.arange(t[0], t[-1] - window_time + 1e-6, window_time)
    
    if len(chunk_starts) == 0:
        chunk_starts = [t[0]]
        window_time = total_time

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
        col_name: round(avg_val, 4),
        "raw_shifts": shifts_array 
    }


def build_shift_matrix(directory_path, target_cycles=15, skip_rows=10, unit="deg"):
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    print(f"Found {len(csv_files)} CSV files. Processing chunks for Welch's T-Test...")
    
    results = [process_file_chunks(f, target_cycles, skip_rows, unit) for f in csv_files]
    results = [r for r in results if r]
            
    if not results: return None, None, None
    
    baselines = {}
    for r in results:
        if r["Stretch"] == 0 and r["Twist"] == 0:
            baselines[r["Frequency (Hz)"]] = r["raw_shifts"]
            
    for r in results:
        freq = r["Frequency (Hz)"]
        base_arr = baselines.get(freq)
        curr_arr = r["raw_shifts"]
        
        if base_arr is not None and len(curr_arr) > 1 and len(base_arr) > 1:
            _, p_val = stats.ttest_ind(curr_arr, base_arr, equal_var=False)
            r["P-Value"] = p_val
        else:
            r["P-Value"] = np.nan
        
    df = pd.DataFrame(results)
    val_col = [col for col in df.columns if "Shift" in col][0]
    
    shift_matrix = df.pivot_table(index=["Stretch", "Twist"], columns="Frequency (Hz)", values=val_col)
    pval_matrix = df.pivot_table(index=["Stretch", "Twist"], columns="Frequency (Hz)", values="P-Value")
    
    # --- FIX: Force p-value matrix to map perfectly to the shift matrix to prevent crashes ---
    pval_matrix = pval_matrix.reindex_like(shift_matrix)
    
    return shift_matrix, pval_matrix, val_col

def plot_matrix(shift_matrix, pval_matrix, val_col, show=True, save_path=None):
    plot_df = shift_matrix.copy()
    plot_df.index = [f"Stretch: {s}, Twist: {t}°" for s, t in shift_matrix.index]
    plot_df.columns = [f"{int(f) if f.is_integer() else f}" for f in plot_df.columns]
    
    text_overlay = []
    for i in range(len(shift_matrix)):
        row = []
        for j in range(len(shift_matrix.columns)):
            val = shift_matrix.iloc[i, j]
            p = pval_matrix.iloc[i, j]
            if pd.isna(val):
                row.append("")
            else:
                stars = "**" if pd.notna(p) and p < 0.01 else "*" if pd.notna(p) and p < 0.05 else ""
                p_str = f"p={p:.3f}" if pd.notna(p) else "p=N/A"
                row.append(f"{val:.4f}{stars}<br>{p_str}")
        text_overlay.append(row)
    
    max_abs_val = np.abs(plot_df.values).max()
    base_title = val_col.split(" (")[0]
    
    fig = px.imshow(
        plot_df,
        labels=dict(x="Frequency (Hz)", y="Sensor State", color=val_col),
        x=plot_df.columns,
        y=plot_df.index,
        aspect="auto",
        color_continuous_scale="RdBu_r", 
        zmin=-max_abs_val,
        zmax=max_abs_val
    )
    
    fig.update_traces(
        text=text_overlay, 
        texttemplate="%{text}", 
        hovertemplate="Freq: %{x}Hz<br>State: %{y}<br><br>Shift: %{z}<br>P-val: %{customdata}"
    )
    fig.data[0].customdata = pval_matrix.values
    
    fig.update_layout(
        title=f"Sensor {base_title} Matrix (*p<0.05, **p<0.01 vs Baseline Stretch 0/Twist 0)",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Sensor State"
    )
    fig.update_xaxes(type='category')
    
    if save_path:
        fig.write_html(save_path)
        print(f"\nSaved interactive heatmap to {save_path}")
        
    if show:
        fig.show()


def main():
    parser = argparse.ArgumentParser(description="Process DAQ CSV files to build a Statistical Shift Matrix.")
    parser.add_argument("input_dir", help="Path to DAQ CSV files.")
    parser.add_argument("--unit", choices=["deg", "rad", "sec"], default="deg")
    parser.add_argument("--target-cycles", type=int, default=15)
    parser.add_argument("--skip-rows", type=int, default=10)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save-plot", nargs="?", const="stat_shift_matrix.html", default=None)
    
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Directory '{args.input_dir}' does not exist.")
        return

    shift_matrix, pval_matrix, val_col = build_shift_matrix(
        args.input_dir, args.target_cycles, args.skip_rows, args.unit
    )

    if shift_matrix is not None:
        print(f"\n--- FINAL MATRIX ({val_col}) ---")
        print(shift_matrix.round(4))
        
        print(f"\n--- P-VALUES (Welch's T-Test vs Stretch 0, Twist 0) ---")
        print(pval_matrix.fillna("N/A").round(4))
        
        if args.plot or args.save_plot:
            plot_matrix(shift_matrix, pval_matrix, val_col, show=args.plot, save_path=args.save_plot)

if __name__ == "__main__":
    main()