"""
cleaning.py - Load a session CSV, clean it and add derived variables.

Pipeline (each step is logged in CleanResult.log so you can show / defend what was changed):
  1. load            read CSV, rename legacy columns, convert text to numbers
  2. duplicates      exact duplicate rows
  3. segments        Teensy restarts (seq goes back) -> segment id; duplicate seq inside a segment
  4. time base       one absolute clock (GPS UTC if available, else Pi clock), elapsed time
  5. gaps            missing packets / time holes (kept as a list, lines are broken there in plots)
  6. ranges          physically impossible values -> NaN (ranges in schema.py)
  7. gps             no fix, few satellites, bad HDOP, position jumps, speed spikes -> NaN
  8. vibration       dominant frequency of a noise-level peak -> NaN
  9. derived         distance, accelerations, total vibration, battery power and energy
Nothing is interpolated: a cleaned value is either the measured one or NaN.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .schema import AXES, COLUMNS, LEGACY_V1


@dataclass
class CleanOptions:
    timezone: str = "America/Mexico_City"   # for displaying GPS UTC as local time
    min_sats: int = 4
    max_hdop: float = 5.0
    max_gps_speed_kmh: float = 250.0         # position jump: implied speed above this = bad fix
    max_long_accel_ms2: float = 12.0         # a car cannot change speed faster than ~1.2 g
    min_vib_amp_ms2: float = 0.05            # dominant frequency below this amplitude is noise
    gap_s: float = 1.5                       # time step above this = gap
    moving_kmh: float = 5.0                  # "moving" threshold for stats


@dataclass
class CleanResult:
    df: pd.DataFrame
    log: list[dict] = field(default_factory=list)
    gaps: pd.DataFrame = field(default_factory=pd.DataFrame)
    meta: dict = field(default_factory=dict)

    def log_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.log, columns=["step", "values_affected", "detail"])


# ----------------------------------------------------------------------------------------------
def read_csv(source) -> pd.DataFrame:
    """source: path, file-like object or bytes."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    df = pd.read_csv(source, comment="#", skipinitialspace=True, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _mask(df: pd.DataFrame, cols, cond, res: CleanResult, step: str, detail: str) -> None:
    """Set df[cols] = NaN where cond is True, and log how many real values were removed."""
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return
    cond = pd.Series(cond, index=df.index).fillna(False).astype(bool)
    n = int(df.loc[cond, cols].notna().sum().sum())
    if n:
        df.loc[cond, cols] = np.nan
    res.log.append({"step": step, "values_affected": n, "detail": detail})


# ----------------------------------------------------------------------------------------------
def clean_session(raw: pd.DataFrame, opt: CleanOptions | None = None, name: str = "") -> CleanResult:
    opt = opt or CleanOptions()
    df = raw.copy()
    res = CleanResult(df=df)
    res.meta["file"] = name
    res.meta["rows_raw"] = len(df)

    # 1. load ---------------------------------------------------------------------------------
    legacy = [c for c in df.columns if c in LEGACY_V1]
    if legacy:
        df = df.rename(columns=LEGACY_V1)
        for g in ("gx", "gy", "gz"):                 # v1 gyro was rad/s, v2 is deg/s
            if g in df.columns:
                df[g] = pd.to_numeric(df[g], errors="coerce") * 57.29578
        res.log.append({"step": "load", "values_affected": len(legacy),
                        "detail": "old v1 format: columns renamed, gyro rad/s -> deg/s"})
    res.meta["format"] = "v1 (legacy)" if legacy else "v2"

    bad_text = 0
    for c in df.columns:
        if c in ("pi_time", "utc_time"):
            continue
        before = df[c].notna().sum()
        df[c] = pd.to_numeric(df[c], errors="coerce")
        bad_text += int(before - df[c].notna().sum())
    res.log.append({"step": "load", "values_affected": bad_text,
                    "detail": "text that is not a number (corrupted cells) -> NaN"})

    if "seq" in df.columns:                           # every real packet has a number
        n0 = len(df)
        df = df.dropna(subset=["seq"]).reset_index(drop=True)
        res.log.append({"step": "load", "values_affected": n0 - len(df),
                        "detail": "rows without a packet number (truncated / corrupted lines) removed"})

    # 2. duplicates -----------------------------------------------------------------------------
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    res.log.append({"step": "duplicates", "values_affected": n0 - len(df),
                    "detail": "identical rows removed"})

    # 3. segments (Teensy restarts) -------------------------------------------------------------
    if "seq" in df.columns:
        restart = df["seq"].diff() < 0
        df["segment"] = restart.cumsum().astype(int)
        res.meta["teensy_restarts"] = int(restart.sum())
        n0 = len(df)
        df = df.drop_duplicates(subset=["segment", "seq"], keep="first").reset_index(drop=True)
        res.log.append({"step": "segments", "values_affected": n0 - len(df),
                        "detail": f"repeated packet numbers removed; {int(restart.sum())} Teensy restart(s)"})
    else:
        df["segment"] = 0
        res.meta["teensy_restarts"] = 0

    # 4. time base ------------------------------------------------------------------------------
    pi_t = pd.to_datetime(df["pi_time"], errors="coerce") if "pi_time" in df.columns else None
    utc_t = None
    if "utc_time" in df.columns:
        utc_t = pd.to_datetime(df["utc_time"], errors="coerce", utc=True)
        if "time_src" in df.columns:
            utc_t = utc_t.where(df["time_src"] >= 1)
        utc_t = utc_t.dt.tz_convert(opt.timezone).dt.tz_localize(None)
    if utc_t is not None and utc_t.notna().mean() >= 0.5:
        t, res.meta["time_source"] = utc_t, "GPS UTC (shown in local time)"
    elif pi_t is not None:
        t, res.meta["time_source"] = pi_t, "Raspberry Pi clock"
    else:
        t, res.meta["time_source"] = pd.Series(pd.NaT, index=df.index), "none"
    t = pd.to_datetime(t)
    # fill holes in the chosen clock with the Teensy microsecond clock (same segment)
    if "t_us" in df.columns and t.isna().any() and t.notna().any():
        t_s = (t - pd.Timestamp("1970-01-01")).dt.total_seconds()
        offset = (t_s - df["t_us"] / 1e6).groupby(df["segment"]).transform("median")
        filled = df["t_us"] / 1e6 + offset
        n_fill = int((t.isna() & filled.notna()).sum())
        t = pd.to_datetime(t_s.fillna(filled), unit="s")
        res.log.append({"step": "time base", "values_affected": n_fill,
                        "detail": "timestamps missing in the clock, rebuilt from the Teensy clock"})
    df["time"] = t
    n0 = len(df)
    df = df.dropna(subset=["time"]).sort_values("time", kind="stable").reset_index(drop=True)
    if n0 - len(df):
        res.log.append({"step": "time base", "values_affected": n0 - len(df),
                        "detail": "rows without any usable time removed"})
    df["elapsed_min"] = (df["time"] - df["time"].min()).dt.total_seconds() / 60.0

    # 5. gaps -----------------------------------------------------------------------------------
    dt = df["time"].diff().dt.total_seconds()
    gap_rows = df.index[dt > opt.gap_s]
    gaps = pd.DataFrame({
        "start": df["time"].shift(1).loc[gap_rows].values,
        "end": df["time"].loc[gap_rows].values,
        "seconds": dt.loc[gap_rows].values,
    })
    lost = 0
    if "seq" in df.columns:
        dseq = df.groupby("segment")["seq"].diff()
        lost = int((dseq[dseq > 1] - 1).sum())
    res.gaps = gaps
    res.meta["lost_packets"] = lost
    res.meta["gaps"] = len(gaps)
    res.log.append({"step": "gaps", "values_affected": len(gaps),
                    "detail": f"time holes > {opt.gap_s:g} s ({lost} packets lost); lines are broken there"})

    # 6. physical ranges ------------------------------------------------------------------------
    for c, spec in COLUMNS.items():
        if c not in df.columns or (spec.lo is None and spec.hi is None):
            continue
        cond = pd.Series(False, index=df.index)
        if spec.lo is not None:
            cond |= df[c] < spec.lo
        if spec.hi is not None:
            cond |= df[c] > spec.hi
        if cond.any():
            _mask(df, [c], cond, res, "ranges", f"{c} outside [{spec.lo}, {spec.hi}] {spec.unit}")

    # 7. GPS ------------------------------------------------------------------------------------
    pos = ["lat", "lon", "alt_m", "speed_kmh", "course_deg"]
    if "lat" in df.columns and "lon" in df.columns:
        no_fix = pd.Series(False, index=df.index)
        if "gps_fix" in df.columns:
            no_fix |= df["gps_fix"] != 1
        if "sats" in df.columns:
            no_fix |= df["sats"] < opt.min_sats
        no_fix |= (df["lat"] == 0) & (df["lon"] == 0)
        _mask(df, pos, no_fix, res, "gps", f"no fix or fewer than {opt.min_sats} satellites")
        if "hdop" in df.columns:
            _mask(df, pos, df["hdop"] > opt.max_hdop, res, "gps", f"HDOP above {opt.max_hdop:g} (poor geometry)")
        # position jumps: a fix that is far from BOTH neighbours (implied speed in and out too high)
        valid = df[["lat", "lon"]].notna().all(axis=1)
        v = df.loc[valid]
        d_in = _haversine_m(v["lat"].shift(), v["lon"].shift(), v["lat"], v["lon"])
        d_out = _haversine_m(v["lat"], v["lon"], v["lat"].shift(-1), v["lon"].shift(-1))
        dt_in = v["time"].diff().dt.total_seconds()
        dt_out = -v["time"].diff(-1).dt.total_seconds()
        lim = opt.max_gps_speed_kmh / 3.6
        bad_in = (d_in / dt_in > lim) & (dt_in > 0)
        bad_out = (d_out / dt_out > lim) & (dt_out > 0)
        jump = pd.Series(False, index=df.index)
        jump.loc[v.index] = bad_in & (bad_out | v["lat"].shift(-1).isna())
        _mask(df, pos, jump, res, "gps", f"position jump (implied > {opt.max_gps_speed_kmh:g} km/h)")
    if "speed_kmh" in df.columns:
        s = df["speed_kmh"] / 3.6
        dts = df["time"].diff().dt.total_seconds()
        a_prev = (s - s.shift()) / dts
        a_next = (s.shift(-1) - s) / df["time"].diff(-1).abs().dt.total_seconds()
        spike = (a_prev.abs() > opt.max_long_accel_ms2) & (a_next.abs() > opt.max_long_accel_ms2) \
            & (np.sign(a_prev) != np.sign(a_next))
        _mask(df, ["speed_kmh"], spike, res, "gps",
              f"speed spike (> {opt.max_long_accel_ms2:g} m/s^2 in and out)")

    # 8. vibration ------------------------------------------------------------------------------
    for a in AXES:
        if f"{a}_freq" in df.columns and f"{a}_amp" in df.columns:
            _mask(df, [f"{a}_freq"], df[f"{a}_amp"] < opt.min_vib_amp_ms2, res, "vibration",
                  f"{a}: dominant frequency of a peak below {opt.min_vib_amp_ms2:g} m/s^2 (noise)")

    # 9. derived --------------------------------------------------------------------------------
    dt = df["time"].diff().dt.total_seconds()
    consecutive = (dt > 0) & (dt <= opt.gap_s)
    if "lat" in df.columns and "lon" in df.columns:
        step_m = _haversine_m(df["lat"].shift(), df["lon"].shift(), df["lat"], df["lon"])
        step_m = step_m.where(consecutive).fillna(0.0)
        df["dist_km"] = step_m.cumsum() / 1000.0
    if "speed_kmh" in df.columns:
        sp = (df["speed_kmh"] / 3.6).rolling(3, center=True, min_periods=2).mean()
        sec = (df["time"] - df["time"].min()).dt.total_seconds()
        df["long_accel_ms2"] = (sp.diff() / sec.diff()).where(consecutive)
        if "gz" in df.columns:   # centripetal a = v * yaw rate (assumes the IMU Z axis is vertical)
            df["lat_accel_ms2"] = df["speed_kmh"] / 3.6 * np.radians(df["gz"])
    rms_cols = [f"{a}_rms" for a in AXES if f"{a}_rms" in df.columns]
    if len(rms_cols) == 3:
        df["vib_rms_total"] = np.sqrt((df[rms_cols] ** 2).sum(axis=1, min_count=3))
    if "batt_v" in df.columns and "batt_a" in df.columns:
        df["batt_w"] = df["batt_v"] * df["batt_a"]
        df["energy_wh"] = (df["batt_w"] * dt.where(consecutive) / 3600.0).fillna(0).cumsum()

    # meta -------------------------------------------------------------------------------------
    res.df = df
    if len(df):
        res.meta["start"] = df["time"].iloc[0]
        res.meta["end"] = df["time"].iloc[-1]
        dur = (df["time"].iloc[-1] - df["time"].iloc[0]).total_seconds()
        res.meta["duration_s"] = dur
        res.meta["rows_clean"] = len(df)
        res.meta["completeness_pct"] = 100.0 * len(df) / max(1.0, dur + 1.0)
    return res


def load_and_clean(source, opt: CleanOptions | None = None, name: str = "") -> CleanResult:
    return clean_session(read_csv(source), opt, name)


def with_gap_breaks(df: pd.DataFrame, gap_s: float = 1.5) -> pd.DataFrame:
    """Insert one empty row inside every time hole so plot lines do not bridge missing data."""
    if df.empty:
        return df
    dt = df["time"].diff().dt.total_seconds()
    idx = df.index[dt > gap_s]
    if len(idx) == 0:
        return df
    breaks = pd.DataFrame({"time": df["time"].loc[idx] - pd.Timedelta(milliseconds=1)})
    if "elapsed_min" in df.columns:
        breaks["elapsed_min"] = df["elapsed_min"].loc[idx] - 1e-6
    return pd.concat([df, breaks], ignore_index=True).sort_values("time", kind="stable").reset_index(drop=True)


def read_raw(source) -> pd.DataFrame | None:
    """Optional session_*_raw.csv (512 samples/s). Returns columns t_s, ax, ay, az in m/s^2."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    if hasattr(source, "readline"):
        first = source.readline()
        source.seek(0)
    else:
        with open(source, "rb") as fh:
            first = fh.readline()
    line = first.decode("ascii", "replace") if isinstance(first, bytes) else first
    lsb = 8192.0
    for tok in line.replace("#", " ").split():
        if tok.startswith("acc_lsb_per_g="):
            lsb = float(tok.split("=", 1)[1])
    df = pd.read_csv(source, comment="#")
    if not {"t_us", "ax_cnt", "ay_cnt", "az_cnt"}.issubset(df.columns):
        return None
    k = 9.80665 / lsb
    out = pd.DataFrame({"t_s": (df["t_us"] - df["t_us"].iloc[0]) / 1e6})
    for a in AXES:
        out[a] = df[f"{a}_cnt"] * k
    out["seq"] = df["seq"]
    return out
