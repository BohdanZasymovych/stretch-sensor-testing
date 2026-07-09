import os
import sys
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import minimize_scalar


def fit_sine(t, y):
    """
    Fits a sine wave using linear least squares and a bounded frequency search.
    This completely prevents 'amplitude collapse' on long, high-frequency datasets.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    print(len(t))
    
    offset = np.mean(y)
    N = len(t)
    dt = t[1] - t[0]
    yf = np.fft.fft(y - offset)
    xf = np.fft.fftfreq(N, dt)
    
    idx = np.argmax(np.abs(yf[1:N//2])) + 1
    f_rough = xf[idx]
    
    def negative_amplitude(f):
        """Returns the negative amplitude for a given frequency to find the maximum."""
        w = 2 * np.pi * f
        X = np.column_stack([np.sin(w * t), np.cos(w * t), np.ones_like(t)])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            return -np.sqrt(coeffs[0]**2 + coeffs[1]**2)
        except np.linalg.LinAlgError:
            return 0.0


    res = minimize_scalar(
        negative_amplitude, 
        bounds=(f_rough - 1.0, f_rough + 1.0), 
        method='bounded'
    )
    f_exact = res.x
    
    w_exact = 2 * np.pi * f_exact
    X = np.column_stack([np.sin(w_exact * t), np.cos(w_exact * t), np.ones_like(t)])
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    
    a, b, C = coeffs
    fit_A = np.sqrt(a**2 + b**2)
    fit_phi = np.arctan2(b, a) 
    fit_C = C
    
    fitted_func = lambda t_val, w=w_exact, p=fit_phi, amp=fit_A, c=fit_C: amp * np.sin(w * t_val + p) + c
    
    return {
        "amplitude": float(fit_A),
        "phase": float(fit_phi),
        "offset": float(fit_C),
        "freq": float(f_exact),
        "func": fitted_func
    }


def plot_csv_interactive(csv_path, skip_rows=10, curve_points=20000,
                          output_html=None, open_browser=True):
    """Read a DAQ CSV, fit both channels with fit_sine(), and render an
    interactive plot with four independently-toggleable traces."""
    
    df = pd.read_csv(csv_path)
    required_cols = {"timestamp", "voltage_initial_mv", "voltage_distorted_mv"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {required_cols}")

    df = df.iloc[skip_rows:].reset_index(drop=True)

    t = df["timestamp"].to_numpy(dtype=float)
    initial = df["voltage_initial_mv"].to_numpy(dtype=float)
    distorted = df["voltage_distorted_mv"].to_numpy(dtype=float)

    fit_initial = fit_sine(t, initial)
    fit_distorted = fit_sine(t, distorted)

    t_smooth = np.linspace(t.min(), t.max(), curve_points)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t, y=initial, mode="markers", name="Initial signal (points)",
        marker=dict(size=4, color="royalblue", opacity=0.6),
    ))
    fig.add_trace(go.Scatter(
        x=t, y=distorted, mode="markers", name="Distorted signal (points)",
        marker=dict(size=4, color="firebrick", opacity=0.6),
    ))
    
    fig.add_trace(go.Scatter(
        x=t_smooth, y=fit_initial["func"](t_smooth), mode="lines",
        name="Initial signal (fit)", line=dict(color="royalblue", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=t_smooth, y=fit_distorted["func"](t_smooth), mode="lines",
        name="Distorted signal (fit)", line=dict(color="firebrick", width=2, dash="dash"),
    ))

    display_freq = fit_initial['freq']

    fig.update_layout(
        title=(f"{os.path.basename(csv_path)}  |  Auto-detected Freq: {display_freq:.2f} Hz  "
               f"(click legend entries to show/hide traces)"),
        xaxis_title="Time (s)",
        yaxis_title="Voltage (mV)",
        legend=dict(itemclick="toggle", itemdoubleclick="toggleothers"),
        template="plotly_white",
        hovermode="closest",
    )

    if output_html is None:
        base, _ = os.path.splitext(csv_path)
        output_html = base + "_fit.html"
    fig.write_html(output_html)
    print(f"Saved interactive plot to {output_html}")

    if open_browser:
        try:
            fig.show()
        except Exception as e:
            print(f"(Could not auto-open a browser: {e}. "
                  f"Open {output_html} manually instead.)", file=sys.stderr)

    return fig, fit_initial, fit_distorted


def main():
    parser = argparse.ArgumentParser(description="Plot and fit DAQ sine wave data.",
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", help="Path to the DAQ CSV file to plot")
    parser.add_argument("--skip-rows", type=int, default=10,
                         help="Startup rows to ignore (default: 10)")
    parser.add_argument("--no-browser", dest="open_browser", action="store_false",
                         default=True, help="Don't attempt to open a browser window; "
                                             "just save the HTML file")
    args = parser.parse_args()

    fig, fit_i, fit_d = plot_csv_interactive(
        args.csv_path, skip_rows=args.skip_rows, open_browser=args.open_browser
    )

    print(f"\nInitial signal fit:   freq={fit_i['freq']:.4f} Hz, amplitude={fit_i['amplitude']:.2f} mV, "
          f"phase={fit_i['phase']:.4f} rad, offset={fit_i['offset']:.2f} mV")
    print(f"Distorted signal fit: freq={fit_d['freq']:.4f} Hz, amplitude={fit_d['amplitude']:.2f} mV, "
          f"phase={fit_d['phase']:.4f} rad, offset={fit_d['offset']:.2f} mV")

if __name__ == "__main__":
    main()