import csv
import argparse
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class DataLoader:
    def __init__(self, filename: str):
        self.filename = filename

    def load(self, start_sec: float = None, end_sec: float = None):
        times = []
        voltages = []
        t0 = None

        try:
            with open(self.filename, mode="r", encoding="utf-8") as file:
                csv_reader = csv.reader(file)
                
                next(csv_reader, None)

                for row in csv_reader:
                    if not row or len(row) < 2:
                        continue

                    try:
                        timestamp = float(row[0])
                        voltage = float(row[1])
                    except ValueError:
                        continue

                    if t0 is None:
                        t0 = timestamp
                    
                    rel_time = timestamp - t0

                    if start_sec is not None and rel_time < start_sec:
                        continue
                    if end_sec is not None and rel_time > end_sec:
                        continue

                    times.append(rel_time)
                    voltages.append(voltage)
                    
        except FileNotFoundError:
            print(f"Error: Could not find the file '{self.filename}'.")
            sys.exit(1)

        return times, voltages

class PlotGenerator:
    @staticmethod
    def show_plot(times: list, voltages: list, filename: str, save_path: str = None):
        if not times or not voltages:
            print("No data found in the specified time range.")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(times, voltages, label="Voltage", color="#2ca02c", linewidth=1.5)
        
        plt.title(f"Sensor Data: {filename}")
        plt.xlabel("Time (Seconds from start)")
        plt.ylabel("Voltage (mV)")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            
        plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description="Plot saved serial voltage data from a CSV.")

    parser.add_argument("--file", "-f", type=str, required=True)
    parser.add_argument("--start", "-s", type=float, default=None)
    parser.add_argument("--end", "-e", type=float, default=None)
    parser.add_argument("--save", "-o", type=str, default=None)

    return parser.parse_args()

def main():
    args = parse_args()

    if args.start is not None and args.end is not None and args.start >= args.end:
        print("Error: --start time must be less than --end time.")
        sys.exit(1)

    print(f"Reading data from {args.file}...")
    loader = DataLoader(args.file)
    times, voltages = loader.load(start_sec=args.start, end_sec=args.end)

    print(f"Loaded {len(times)} data points. Generating plot...")
    plotter = PlotGenerator()
    plotter.show_plot(times, voltages, args.file, args.save)

if __name__ == "__main__":
    main()