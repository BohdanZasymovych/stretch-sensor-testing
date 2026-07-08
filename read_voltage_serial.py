import time
import csv
import serial
import argparse
import matplotlib.pyplot as plt
from collections import deque
from abc import ABC, abstractmethod

# --- Constants for ADC Conversion ---
REFERENCE_HIGH_V = 5.0
ADC_RESOLUTION = 1023.0


class DataProcessor(ABC):
    """Base class for anything that handles incoming serial data."""
    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def process(self, timestamp: float, voltage_initial: float, voltage_distorted: float):
        pass

    @abstractmethod
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
        self.csv_writer.writerow(["timestamp", "voltage_initial_mv", "voltage_distorted_mv"])
        print(f"Logging data to {self.filename}")

    def process(self, timestamp: float, voltage_initial: float, voltage_distorted: float):
        self.csv_writer.writerow([timestamp, voltage_initial, voltage_distorted])

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
        self.voltages_initial = deque(maxlen=self.window_size)
        self.voltages_distorted = deque(maxlen=self.window_size)

        self.fig = None
        self.ax = None
        self.line_initial = None
        self.line_distorted = None

    def setup(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.line_initial, = self.ax.plot([], [], 'b-', linewidth=2, label="Initial")
        self.line_distorted, = self.ax.plot([], [], 'r-', linewidth=2, label="Distorted")

        self.ax.set_title("Real-Time Sensor Voltage")
        self.ax.set_xlabel("Time (seconds)")
        self.ax.set_ylabel("Voltage (mV)")
        self.ax.set_ylim(0, 6000)
        self.ax.grid(True)
        self.ax.legend(loc="upper right")

    def process(self, timestamp: float, voltage_initial: float, voltage_distorted: float):
        if self.start_time is None:
            self.start_time = timestamp

        rel_time = timestamp - self.start_time

        self.timestamps.append(rel_time)
        self.voltages_initial.append(voltage_initial)
        self.voltages_distorted.append(voltage_distorted)

        self.point_count += 1

        if self.point_count % self.update_interval == 0:
            self.line_initial.set_data(self.timestamps, self.voltages_initial)
            self.line_distorted.set_data(self.timestamps, self.voltages_distorted)
            self.ax.relim()
            self.ax.autoscale_view(scaley=False)
            plt.pause(0.001)

    def cleanup(self):
        print("Freezing plot. Close the plot window to exit completely.")
        plt.ioff()
        plt.show()


class SerialMonitor:
    def __init__(self, port: str, baud_rate: int, timeout_s: int, duration: float = None):
        self.port = port
        self.baud_rate = baud_rate
        self.timeout_s = timeout_s
        self.duration = duration
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
            
            if self.duration is not None:
                print(f"Program will run for {self.duration} seconds.")

            ser.reset_input_buffer()
            ser.readline()

            start_time = time.time()

            while True:
                # Duration is still checked using PC time, which is perfectly fine.
                if self.duration is not None and (time.time() - start_time) >= self.duration:
                    print(f"\nTarget duration of {self.duration} seconds reached. Stopping...")
                    break

                if ser.in_waiting > 0:
                    raw_data = ser.readline().decode("utf-8", errors="ignore").strip()

                    try:
                        parts = raw_data.split(",")
                        # We now expect 3 fields: micros, raw1, raw2
                        if len(parts) != 3:
                            continue

                        # 1. Parse time from Arduino (microseconds -> seconds)
                        timestamp = float(parts[0]) / 1_000_000.0
                        
                        # 2. Parse raw ADC integers
                        raw_initial = int(parts[1])
                        raw_distorted = int(parts[2])

                        # 3. Calculate voltages in millivolts
                        voltage_initial = (raw_initial / ADC_RESOLUTION) * REFERENCE_HIGH_V * 1000.0
                        voltage_distorted = (raw_distorted / ADC_RESOLUTION) * REFERENCE_HIGH_V * 1000.0

                        for p in self.processors:
                            p.process(timestamp, voltage_initial, voltage_distorted)

                    except ValueError:
                        continue

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
    parser.add_argument("--baud_rate", "-r", type=int, default=2000000, help="Baud rate")
    parser.add_argument("--timeout", "-t", type=int, default=1, help="Serial timeout")
    parser.add_argument("--file", "-f", type=str, default=None, help="Save data to the specified CSV file")
    parser.add_argument("--plot", action="store_true", help="Display a real-time live plot of the data")    
    parser.add_argument("--duration", "-d", type=float, default=None, help="Time in seconds to log data before exiting")

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.file and not args.plot:
        print("Notice: No --file or --plot flags provided. Data will only be printed to the console.")

    monitor = SerialMonitor(args.port, args.baud_rate, args.timeout, duration=args.duration)

    if args.file:
        monitor.add_processor(CSVLogger(args.file))

    if args.plot:
        monitor.add_processor(RealTimePlotter(window_size=200, update_interval=10))

    monitor.run()


if __name__ == "__main__":
    main()