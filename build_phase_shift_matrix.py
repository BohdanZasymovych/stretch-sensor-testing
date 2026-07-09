import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import plotly.express as px

# ------------------------------------------------------------------ #
# 1. The Core Math
# ------------------------------------------------------------------ #

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
            
            if not np.isfinite(f_rough):
                f_rough = 1.0
            
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
    """Safely averages angles to prevent wrapping errors."""
    complex_vectors = np.exp(1j * np.array(angles_rad))
    return np.angle(np.mean(complex_vectors))

# ------------------------------------------------------------------ #
# 2. The Chunking Logic
# ------------------------------------------------------------------ #

def process_file_chunks(csv_path, target_cycles=15, skip_rows=10, unit="deg"):
    """
    Reads a CSV, splits it into dynamic chunks, calculates the 
    shift for each chunk, and returns the circular mean in the requested unit.
    """
    name = os.path.basename(csv_path)
    match = re.search(r"stretch(\d+)(?:_twist(\d+))?_sine_(\d+)hz", name, re.IGNORECASE)
    if not match:
        return None
    
    stretch = int(match.group(1))
    twist = int(match.group(2)) if match.group(2) else 0
    nominal_freq = float(match.group(3))
    
    if twist == 360:
        return None
    
    df = pd.read_csv(csv_path).iloc[skip_rows:].reset_index(drop=True)
    t = df["timestamp"].values
    initial = df["voltage_initial_mv"].values
    distorted = df["voltage_distorted_mv"].values
    
    total_time = t[-1] - t[0]
    window_time = target_cycles / nominal_freq
    if window_time > total_time:
        window_time = total_time
        
    chunk_starts = np.arange(t[0], t[-1] - window_time, window_time)
    
    if len(chunk_starts) == 0:
        chunk_starts = [t[0]]
        window_time = total_time

    phase_shifts_rad = []
    
    for start in chunk_starts:
        mask = (t >= start) & (t < start + window_time)
        t_chunk = t[mask]
        i_chunk = initial[mask]
        d_chunk = distorted[mask]
        
        if len(t_chunk) < 10: 
            continue
            
        fit_i = fit_sine(t_chunk, i_chunk)
        chunk_freq = fit_i["freq"]
        
        fit_d = fit_sine(t_chunk, d_chunk, forced_freq=chunk_freq)
        
        delta_phi = fit_d["phase"] - fit_i["phase"]
        phase_shifts_rad.append(delta_phi)

    if not phase_shifts_rad:
        return None

    avg_shift_rad = circular_mean(phase_shifts_rad)
    
    if unit == "sec":
        final_val = avg_shift_rad / (2 * np.pi * nominal_freq)
        col_name = "Time Shift (s)"
    elif unit == "rad":
        final_val = avg_shift_rad
        col_name = "Phase Shift (rad)"
    else:
        final_val = np.degrees(avg_shift_rad)
        col_name = "Phase Shift (deg)"
    
    return {
        "Stretch": stretch,
        "Twist": twist,
        "Frequency (Hz)": nominal_freq,
        col_name: round(final_val, 4)
    }

# ------------------------------------------------------------------ #
# 3. Matrix Builder & Plotter
# ------------------------------------------------------------------ #

def build_shift_matrix(directory_path, target_cycles=15, skip_rows=10, unit="deg"):
    """Scans directory and builds a Pandas pivot table."""
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    print(f"Found {len(csv_files)} CSV files. Processing chunks (Target Cycles: {target_cycles}, Skip Rows: {skip_rows})...")
    
    results = []
    for file in csv_files:
        res = process_file_chunks(file, target_cycles=target_cycles, skip_rows=skip_rows, unit=unit)
        if res:
            results.append(res)
            
    if not results:
        print("No valid files found or processed.")
        return None, None
        
    df = pd.DataFrame(results)
    val_col = [col for col in df.columns if "Shift" in col][0]
    
    matrix = df.pivot_table(
        index=["Stretch", "Twist"], 
        columns="Frequency (Hz)", 
        values=val_col
    )
    
    return matrix, val_col

def plot_matrix(matrix, val_col, show=True, save_path=None):
    """Generates an interactive Plotly heatmap of the dynamic shift matrix."""
    plot_df = matrix.copy()
    plot_df.index = [f"Stretch: {s}, Twist: {t}°" for s, t in matrix.index]
    
    # --- VISUAL FIX: Force columns to be strings so they are equally wide ---
    plot_df.columns = [f"{int(f) if f.is_integer() else f}" for f in plot_df.columns]
    
    max_abs_val = np.abs(plot_df.values).max()
    base_title = val_col.split(" (")[0]
    
    fig = px.imshow(
        plot_df,
        labels=dict(x="Frequency (Hz)", y="Sensor State (Stretch & Twist)", color=val_col),
        x=plot_df.columns,
        y=plot_df.index,
        text_auto=".4f", 
        aspect="auto",
        color_continuous_scale="RdBu_r", 
        zmin=-max_abs_val,
        zmax=max_abs_val
    )
    
    fig.update_layout(
        title=f"Sensor {base_title} Matrix",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Sensor State"
    )
    
    # Explicitly tell Plotly the X-axis is categorical, not continuous
    fig.update_xaxes(type='category')
    
    if save_path:
        fig.write_html(save_path)
        print(f"\nSaved interactive heatmap to {save_path}")
        
    if show:
        fig.show()

# ------------------------------------------------------------------ #
# 4. CLI Entry Point
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Process DAQ CSV files to build and plot a dynamic Shift Matrix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input_dir", 
        help="Path to the directory containing the DAQ CSV files."
    )
    parser.add_argument(
        "--unit", 
        choices=["deg", "rad", "sec"], 
        default="deg",
        help="Unit for the calculation: degrees (deg), radians (rad), or seconds (sec)."
    )
    parser.add_argument(
        "--target-cycles", 
        type=int, 
        default=15,
        help="Number of sine wave cycles per calculation chunk."
    )
    parser.add_argument(
        "--skip-rows", 
        type=int, 
        default=10,
        help="Number of initial rows to skip to avoid startup noise."
    )
    parser.add_argument(
        "--plot", 
        action="store_true",
        help="Open an interactive heatmap of the matrix in your browser."
    )
    parser.add_argument(
        "--save-plot", 
        nargs="?",
        const="shift_matrix.html",
        default=None,
        help="Save the heatmap as an HTML file. Optionally provide a filename."
    )
    
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Directory '{args.input_dir}' does not exist.")
        return

    shift_matrix, val_col = build_shift_matrix(
        args.input_dir, 
        target_cycles=args.target_cycles, 
        skip_rows=args.skip_rows,
        unit=args.unit
    )
    
    if shift_matrix is not None:
        print(f"\n--- FINAL MATRIX ({val_col}) ---")
        print(shift_matrix.round(4))
        
        if args.plot or args.save_plot:
            plot_matrix(shift_matrix, val_col, show=args.plot, save_path=args.save_plot)
    else:
        print("\nCould not build matrix. Check if the directory contains valid matching CSV files.")

if __name__ == "__main__":
    main()