"""
=============================================================================
 Arduino Serial Communication  —  serial_comms.py
=============================================================================
Handles bidirectional USB-Serial communication with the Arduino.

Protocol (simple text lines terminated with '\\n'):
  Arduino → Pi:
    SENSORS:S1=x,S2=x,S3=x,S4=x,DIST=xx   (sensor data)
    IMAGE_READY                              (camera pointed at card — classify now)
    OBSTACLE_DETECTED                        (obstacle seen, about to bypass)
    PI_TIMEOUT                               (Pi didn't respond in time)

  Pi → Arduino:
    DONE                                     (classification finished, continue)
    START_RUN                                (start the run)
    STOP                                     (emergency stop)

Adjust SERIAL_PORT and SERIAL_BAUD in config.py to match your setup.
=============================================================================
"""

import threading
import time
import serial
import config


class ArduinoComm:
    """Thread-safe serial wrapper for Arduino communication."""

    def __init__(
        self,
        port: str = config.SERIAL_PORT,
        baud: int = config.SERIAL_BAUD,
        timeout: float = config.SERIAL_TIMEOUT,
    ):
        self._port = port
        self._baud = baud
        self._timeout = timeout

        self._ser = None
        self._lock = threading.Lock()

        # Latest sensor payload received from the Arduino
        self._last_sensor_data = {}
        self._running = False

        # Event callback — set by app.py to handle IMAGE_READY etc.
        self._event_callback = None

    def set_event_callback(self, callback):
        """Register a callback: callback(event_name: str) called on Arduino events."""
        self._event_callback = callback

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def connect(self):
        """Open the serial port.  Call this once at startup."""
        try:
            self._ser = serial.Serial(
                port=self._port,
                baudrate=self._baud,
                timeout=self._timeout,
            )
            time.sleep(2)  # Arduino resets on serial open — wait for boot
            print(f"[SERIAL] Connected to {self._port} @ {self._baud}")
        except serial.SerialException as exc:
            print(f"[SERIAL] Could not open {self._port}: {exc}")
            self._ser = None

    def disconnect(self):
        self._running = False
        if self._ser and self._ser.is_open:
            self._ser.close()
            print("[SERIAL] Disconnected.")

    @property
    def is_connected(self):
        return self._ser is not None and self._ser.is_open

    # ------------------------------------------------------------------
    # Sending commands  (Pi → Arduino)
    # ------------------------------------------------------------------
    def send(self, command: str):
        """Send a text command to the Arduino (newline appended automatically)."""
        if not self.is_connected:
            print("[SERIAL] Not connected — cannot send.")
            return False
        with self._lock:
            try:
                self._ser.write(f"{command}\n".encode("utf-8"))
                self._ser.flush()
                return True
            except serial.SerialException as exc:
                print(f"[SERIAL] Write error: {exc}")
                return False

    # Convenience helpers
    def start_run(self):
        return self.send("START_RUN")

    def stop(self):
        return self.send("STOP")

    def done(self):
        """Tell Arduino classification is done, continue bypass."""
        return self.send("DONE")

    # ------------------------------------------------------------------
    # Reading sensor data  (Arduino → Pi) — background thread
    # ------------------------------------------------------------------
    def start_listener(self):
        """Start a daemon thread that continuously reads lines from Arduino."""
        self._running = True
        t = threading.Thread(target=self._listen_loop, daemon=True)
        t.start()

    def _listen_loop(self):
        while self._running and self.is_connected:
            try:
                raw = self._ser.readline().decode("utf-8", errors="replace").strip()
                if raw:
                    self._parse_line(raw)
            except serial.SerialException:
                break
            except Exception as exc:
                print(f"[SERIAL] Read error: {exc}")

    def _parse_line(self, line: str):
        """
        Parse incoming Arduino data.
        Handles:
          SENSORS:S1=x,S2=x,S3=x,S4=x,DIST=xx  — sensor state
          IMAGE_READY                            — camera pointed at card
          OBSTACLE_DETECTED                      — obstacle seen
          PI_TIMEOUT                             — Pi was too slow
        """
        if line.startswith("SENSORS:"):
            payload = line[len("SENSORS:"):]
            data = {}
            for pair in payload.split(","):
                if "=" in pair:
                    key, val = pair.split("=", 1)
                    data[key.strip()] = val.strip()
            with self._lock:
                self._last_sensor_data = data

        elif line in ("IMAGE_READY", "OBSTACLE_DETECTED", "PI_TIMEOUT"):
            print(f"[SERIAL] Event: {line}")
            if self._event_callback:
                # Run callback in a separate thread to avoid blocking the listener
                threading.Thread(
                    target=self._event_callback,
                    args=(line,),
                    daemon=True,
                ).start()

    def get_sensor_data(self):
        """Return the latest parsed sensor dict (thread-safe)."""
        with self._lock:
            return dict(self._last_sensor_data)
