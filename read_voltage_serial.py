import time
import csv
import serial
import argparse
import matplotlib.pyplot as plt
from collections import deque


class DataProcessor:
    """Base class for anything that handles incoming serial data."""
    def setup(self):
        pass

    def process(self, timestamp: float, voltage: float):
        pass

    def cleanup(self):
        pass


class CSVLogger(DataProcessor):
    def __init__(self, filename: str):
        self.filename = filename
        self.file = None
        self.csv_writer = None

    def setup(self):
        self.file = open(self.filename, mode="w", encoding="utf-8", newline="")
        self.csv_writer = csv.writer(self.file)
        self.csv_writer.writerow(["timestamp", "voltage_mv"])
        print(f"Logging data to {self.filename}")

    def process(self, timestamp: float, voltage: float):
        self.csv_writer.writerow([timestamp, voltage])

    def cleanup(self):
        if self.file:
            self.file.close()


class RealTimePlotter(DataProcessor):
    def __init__(self, window_size: int = 200, update_interval: int = 10):
        self.window_size = window_size
        self.update_interval = update_interval
        self.point_count = 0
        self.start_time = None
        
        self.timestamps = deque(maxlen=self.window_size)
        self.voltages = deque(maxlen=self.window_size)
        
        self.fig = None
        self.ax = None
        self.line = None

    def setup(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'b-', linewidth=2)
        
        self.ax.set_title("Real-Time Sensor Voltage")
        self.ax.set_xlabel("Time (seconds)")
        self.ax.set_ylabel("Voltage")
        self.ax.grid(True)

    def process(self, timestamp: float, voltage: float):
        if self.start_time is None:
            self.start_time = timestamp
            
        rel_time = timestamp - self.start_time
        
        self.timestamps.append(rel_time)
        self.voltages.append(voltage)
        
        self.point_count += 1
        
        if self.point_count % self.update_interval == 0:
            self.line.set_data(self.timestamps, self.voltages)
            self.ax.relim()
            self.ax.autoscale_view()
            plt.pause(0.001)

    def cleanup(self):
        print("Freezing plot. Close the plot window to exit completely.")
        plt.ioff()
        plt.show()


class SerialMonitor:
    def __init__(self, port: str, baud_rate: int, timeout_s: int):
        self.port = port
        self.baud_rate = baud_rate
        self.timeout_s = timeout_s
        self.processors = []

    def add_processor(self, processor: DataProcessor):
        self.processors.append(processor)

    def run(self):
        try:
            for p in self.processors:
                p.setup()

            ser = serial.Serial(self.port, self.baud_rate, timeout=self.timeout_s)
            time.sleep(2)
            print(f"\nListening on {self.port} at {self.baud_rate} baud...")
            
            ser.reset_input_buffer()
            ser.readline()

            while True:
                if ser.in_waiting > 0:
                    raw_data = ser.readline().decode("utf-8", errors="ignore").strip()
                    timestamp = time.time()
                    
                    try:
                        voltage = float(raw_data)
                        print(f"Voltage: {voltage:.2f}, Time: {timestamp:.2f}")
                        
                        for p in self.processors:
                            p.process(timestamp, voltage)
                            
                    except ValueError:
                        print(f"Skipped garbled data: '{raw_data}'")

        except serial.SerialException as e:
            print(f"Error opening serial port: {e}")
            print("Is the device plugged in and not open in another program?")
        except KeyboardInterrupt:
            print("\nCtrl+C detected. Closing connection...")
        finally:
            if "ser" in locals() and ser.is_open:
                ser.close()
                print(f"Port {self.port} closed")
            
            for p in self.processors:
                p.cleanup()


def parse_args():
    parser = argparse.ArgumentParser(description="Read, log, and plot serial voltage data.")

    parser.add_argument("--port", "-p", type=str, default="/dev/ttyACM0", help="Serial port name")
    parser.add_argument("--baud_rate", "-r", type=int, default=9600, help="Baud rate")
    parser.add_argument("--timeout", "-t", type=int, default=1, help="Serial timeout")

    parser.add_argument("--file", "-f", type=str, default=None, help="Save data to the specified CSV file")
    parser.add_argument("--plot", action="store_true", help="Display a real-time live plot of the data")

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.file and not args.plot:
        print("Notice: No --file or --plot flags provided. Data will only be printed to the console.")

    monitor = SerialMonitor(args.port, args.baud_rate, args.timeout)

    if args.file:
        monitor.add_processor(CSVLogger(args.file))

    if args.plot:
        monitor.add_processor(RealTimePlotter(window_size=200, update_interval=10))

    monitor.run()


if __name__ == "__main__":
    main()