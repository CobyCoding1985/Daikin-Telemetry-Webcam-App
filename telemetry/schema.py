"""
schema.py - What every CSV column means: label, unit, group and physically valid range.

Edit this file when the hardware changes (new sensors = new rows here). Cleaning, plots and
the summary all read from it, so nothing else has to know the column list.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Col:
    label: str
    unit: str = ""
    group: str = "other"
    lo: float | None = None      # values outside [lo, hi] are physically impossible -> NaN
    hi: float | None = None


COLUMNS: dict[str, Col] = {
    # --- packet / timing (Teensy) ---
    "seq":            Col("Packet number", "", "quality"),
    "t_us":           Col("Teensy time", "us", "quality"),
    "utc_ms":         Col("UTC time", "ms", "quality"),
    "time_src":       Col("Time source", "", "quality", 0, 2),
    "fs_hz":          Col("Sampling rate", "Hz", "quality"),
    "n":              Col("Samples per window", "", "quality"),
    "missed":         Col("Missed samples", "", "quality", 0, None),
    "jitter_max_us":  Col("Max sampling jitter", "us", "quality", 0, None),
    "imu_err":        Col("IMU read errors", "", "quality", 0, None),
    "job_max_us":     Col("Longest job", "us", "quality", 0, None),
    "teensy_dropped": Col("Teensy dropped lines", "", "quality", 0, None),
    # --- vibration (MPU6050 now, ADXL345 later) ---
    "ax_mean": Col("Accel X mean", "m/s^2", "vibration", -40, 40),
    "ay_mean": Col("Accel Y mean", "m/s^2", "vibration", -40, 40),
    "az_mean": Col("Accel Z mean", "m/s^2", "vibration", -40, 40),
    "ax_rms":  Col("Vibration RMS X", "m/s^2", "vibration", 0, 40),
    "ay_rms":  Col("Vibration RMS Y", "m/s^2", "vibration", 0, 40),
    "az_rms":  Col("Vibration RMS Z", "m/s^2", "vibration", 0, 40),
    "ax_peak": Col("Vibration peak X", "m/s^2", "vibration", 0, 80),
    "ay_peak": Col("Vibration peak Y", "m/s^2", "vibration", 0, 80),
    "az_peak": Col("Vibration peak Z", "m/s^2", "vibration", 0, 80),
    "ax_freq": Col("Dominant freq X", "Hz", "vibration", 0, 1000),
    "ay_freq": Col("Dominant freq Y", "Hz", "vibration", 0, 1000),
    "az_freq": Col("Dominant freq Z", "Hz", "vibration", 0, 1000),
    "ax_amp":  Col("Peak amplitude X", "m/s^2", "vibration", 0, 80),
    "ay_amp":  Col("Peak amplitude Y", "m/s^2", "vibration", 0, 80),
    "az_amp":  Col("Peak amplitude Z", "m/s^2", "vibration", 0, 80),
    "gx": Col("Gyro X", "deg/s", "vibration", -500, 500),
    "gy": Col("Gyro Y", "deg/s", "vibration", -500, 500),
    "gz": Col("Gyro Z (yaw rate)", "deg/s", "vibration", -500, 500),
    "imu_temp": Col("IMU temperature", "degC", "environment", -40, 100),
    # --- environment (BME280) ---
    "bme_temp":    Col("Air temperature", "degC", "environment", -20, 70),
    "bme_hum":     Col("Relative humidity", "%", "environment", 0, 100),
    "bme_press":   Col("Pressure", "hPa", "environment", 600, 1100),
    "air_density": Col("Air density", "kg/m^3", "environment", 0.6, 1.5),
    # --- GPS ---
    "lat":        Col("Latitude", "deg", "gps", -90, 90),
    "lon":        Col("Longitude", "deg", "gps", -180, 180),
    "alt_m":      Col("Altitude", "m", "gps", -500, 6000),
    "speed_kmh":  Col("GPS speed", "km/h", "gps", 0, 250),
    "course_deg": Col("Course", "deg", "gps", 0, 360),
    "sats":       Col("Satellites", "", "gps", 0, 64),
    "hdop":       Col("HDOP", "", "gps", 0, 100),   # 99.99 = receiver has no fix
    "gps_fix":    Col("GPS fix", "", "gps", 0, 1),
    "gps_age_ms": Col("GPS fix age", "ms", "gps", 0, None),
    # --- load cell (HX711 now, ADS1256 + 4 gauges later) ---
    "load_mean": Col("Load mean", "units", "load"),
    "load_min":  Col("Load min", "units", "load"),
    "load_max":  Col("Load max", "units", "load"),
    "load_n":    Col("Load readings", "/s", "load", 0, None),
    # --- power (Pi + INA226) ---
    "batt_v":       Col("Battery voltage", "V", "power", 0, 60),
    "batt_a":       Col("Battery current", "A", "power", -50, 50),
    "batt_soc":     Col("Battery charge", "%", "power", 0, 100),
    "pi_undervolt": Col("Pi under-voltage", "", "power", 0, 1),
    # --- derived by cleaning.py ---
    "elapsed_min":    Col("Elapsed time", "min", "derived"),
    "dist_km":        Col("Distance", "km", "derived"),
    "long_accel_ms2": Col("Longitudinal accel", "m/s^2", "derived"),
    "lat_accel_ms2":  Col("Lateral accel (est.)", "m/s^2", "derived"),
    "vib_rms_total":  Col("Vibration RMS total", "m/s^2", "derived"),
    "batt_w":         Col("Battery power", "W", "derived"),
    "energy_wh":      Col("Energy used", "Wh", "derived"),
}

# Old CSV from telemetria_4.0.py / v1 Teensy firmware -> current names.
LEGACY_V1 = {
    "Timestamp": "pi_time", "Accel_X": "ax_mean", "Accel_Y": "ay_mean", "Accel_Z": "az_mean",
    "Gyro_X": "gx", "Gyro_Y": "gy", "Gyro_Z": "gz", "Temp_MPU": "imu_temp",
    "Temp_BME": "bme_temp", "Humidity": "bme_hum", "Pressure": "bme_press",
    "Latitude": "lat", "Longitude": "lon", "Speed_kmh": "speed_kmh", "Altitude": "alt_m",
    "Satellites": "sats", "Fix": "gps_fix",
    "Dyn_Accel_X": "ax_peak", "Dyn_Accel_Y": "ay_peak", "Dyn_Accel_Z": "az_peak",
    "FFT_Amp_X": "ax_amp", "FFT_Amp_Y": "ay_amp", "FFT_Amp_Z": "az_amp",
    "Load_Cell_Raw": "load_mean",
}

AXES = ("ax", "ay", "az")


def label(col: str) -> str:
    c = COLUMNS.get(col)
    if c is None:
        return col
    return f"{c.label} ({c.unit})" if c.unit else c.label
