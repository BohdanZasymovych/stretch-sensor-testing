import time
import csv
import serial
import argparse

class SerialLogger:
    FILENAME = "results.csv"
    PORT = "/dev/ttyACM0"
    BAUD_RATE = 9600
    TIMEOUT_S = 1
    CSV_HEADER = ["timestamp", "voltage_mv"]

    def __init__(self, filename: str=None, port: str=None, baud_rate: int=None, timeout_s: int=None):
        self.filename = filename if filename is not None else self.FILENAME
        self.port = port if port is not None else self.PORT
        self.baud_rate = baud_rate if baud_rate is not None else self.BAUD_RATE
        self.timeout_s = timeout_s if timeout_s is not None else self.TIMEOUT_S

    @staticmethod
    def __save_line(timestamp: int, voltage: float, csv_writer):
        csv_writer.writerow([timestamp, voltage])

    def start_logging(self):
        try:
            ser = serial.Serial(self.port, self.baud_rate, timeout=self.timeout_s)
            time.sleep(2)
            print(f"Listening on {self.port} at {self.baud_rate} baud...")

            ser.reset_input_buffer() 
            ser.readline()

            with open(self.filename, mode="w", encoding="utf-8", newline="") as file:
                csv_writer = csv.writer(file)

                csv_writer.writerow(self.CSV_HEADER)

                while True:
                    if ser.in_waiting > 0:
                        raw_voltage = ser.readline().decode("utf-8", errors="ignore").strip()
                        timestamp = time.time()

                        try:
                            voltage = float(raw_voltage)
                            print(f"Voltage: {voltage}, Timestapm: {timestamp}")
                            self.__save_line(timestamp, voltage, csv_writer)
                        except ValueError:
                            print(f"Dropping garbage value: {raw_voltage}")

        except serial.SerialException as e:
            print(f"Error opening serial port: {e}")
            print("Is the device plugged in and not open in another program?")

        except KeyboardInterrupt:
            print("\nClosing connection...")
        
        finally:
            if "ser" in locals() and ser.is_open:
                ser.close()
                print(f"Port {self.port} closed")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", "-f", type=str, default=None, help="Path to the .csv file to save logs")
    parser.add_argument("--port", "-p", type=str, default=None, help="Name of the port to which serial device is connected")
    parser.add_argument("--baud_rate", "-r", type=int, default=None, help="Baud rate of the serial device")
    parser.add_argument("--timeout", "-t", type=int, default=None, help="Timeout to stop reading when there is no data on serial port")

    args = parser.parse_args()

    return args.file, args.port, args.baud_rate, args.timeout

def main():
    file, port, baud_rate, timeout = parse_args()

    serial_logger = SerialLogger(file, port, baud_rate, timeout)

    serial_logger.start_logging()


if __name__ == "__main__":
    main()
