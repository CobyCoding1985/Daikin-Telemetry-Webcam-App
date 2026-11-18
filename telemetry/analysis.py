"""
analysis.py - Numbers that summarise a session (what goes on the first slide of a presentation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cleaning import CleanOptions, CleanResult
from .schema import AXES, label


def _f(x, nd=2):
    """float or None (JSON friendly)."""
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if np.isfinite(x) else None


def _stats(s: pd.Series, nd=2) -> dict:
    s = s.dropna()
    if s.empty:
        return {"mean": None, "min": None, "max": None}
    return {"mean": _f(s.mean(), nd), "min": _f(s.min(), nd), "max": _f(s.max(), nd)}


def summarize(res: CleanResult, opt: CleanOptions | None = None) -> dict:
    opt = opt or CleanOptions()
    df, m = res.df, res.meta
    out: dict = {}

    out["session"] = {
        "file": m.get("file", ""),
        "format": m.get("format"),
        "start": str(m.get("start", "")),
        "end": str(m.get("end", "")),
        "duration_min": _f(m.get("duration_s", 0) / 60.0, 1),
        "rows_raw": m.get("rows_raw"),
        "rows_clean": m.get("rows_clean"),
        "completeness_pct": _f(m.get("completeness_pct"), 1),
        "gaps": m.get("gaps"),
        "lost_packets": m.get("lost_packets"),
        "teensy_restarts": m.get("teensy_restarts"),
        "time_source": m.get("time_source"),
    }

    if "speed_kmh" in df.columns:
        sp = df["speed_kmh"]
        moving = sp > opt.moving_kmh
        dt = df["time"].diff().dt.total_seconds().clip(upper=opt.gap_s).fillna(0)
        out["driving"] = {
            "distance_km": _f(df["dist_km"].iloc[-1], 2) if "dist_km" in df.columns and len(df) else None,
            "max_speed_kmh": _f(sp.max(), 1),
            "mean_speed_moving_kmh": _f(sp[moving].mean(), 1),
            "time_moving_min": _f(dt[moving].sum() / 60.0, 1),
            "time_above_80_min": _f(dt[sp >= 80].sum() / 60.0, 1),
            "gps_valid_pct": _f(100.0 * df["lat"].notna().mean(), 1) if "lat" in df.columns else None,
            "max_accel_ms2": _f(df["long_accel_ms2"].max(), 2) if "long_accel_ms2" in df.columns else None,
            "max_brake_ms2": _f(-df["long_accel_ms2"].min(), 2) if "long_accel_ms2" in df.columns else None,
            "max_lateral_ms2": _f(df["lat_accel_ms2"].abs().max(), 2) if "lat_accel_ms2" in df.columns else None,
        }

    vib = {}
    for a in AXES:
        if f"{a}_rms" not in df.columns:
            continue
        freq = df.get(f"{a}_freq")
        common = None
        if freq is not None and freq.notna().any():
            common = _f(freq.dropna().round(0).mode().iloc[0], 0)
        vib[a] = {"rms": _stats(df[f"{a}_rms"], 3), "peak_max": _f(df.get(f"{a}_peak", pd.Series()).max(), 3),
                  "most_common_freq_hz": common}
    if "vib_rms_total" in df.columns and "speed_kmh" in df.columns:
        both = df[["vib_rms_total", "speed_kmh"]].dropna()
        vib["corr_rms_total_vs_speed"] = _f(both.corr().iloc[0, 1], 3) if len(both) > 10 else None
    if vib:
        out["vibration"] = vib

    env = {c: _stats(df[c]) for c in ("bme_temp", "bme_hum", "bme_press", "air_density") if c in df.columns}
    if env:
        out["environment"] = env
    if "airspeed_kmh" in df.columns and df["airspeed_kmh"].notna().any():
        air = {"max_airspeed_kmh": _f(df["airspeed_kmh"].max(), 1)}
        if "wind_kmh" in df.columns and "speed_kmh" in df.columns:
            fast = df["speed_kmh"] > 40
            air["mean_headwind_kmh_above_40"] = _f(df.loc[fast, "wind_kmh"].mean(), 1)
        if "q_gps_pa" in df.columns:
            both = df[["dp_pa", "q_gps_pa"]].dropna()
            both = both[both["q_gps_pa"] > 20]
            if len(both) > 10:
                air["corr_dp_vs_q_gps"] = _f(both.corr().iloc[0, 1], 3)
                air["dp_over_q_gps"] = _f((both["dp_pa"] / both["q_gps_pa"]).median(), 3)
        out["airspeed"] = air

    if "res_level" in df.columns:
        st = {"warning_s": int((df["res_level"] == 2).sum()), "caution_s": int((df["res_level"] == 1).sum())}
        hot = df.loc[df["res_level"] >= 1, "res_freq_hz"].dropna()
        st["resonance_freq_hz"] = _f(hot.round(0).mode().iloc[0], 0) if len(hot) else None
        st["natural_freq_hz"] = _f(df["res_fn_hz"].dropna().iloc[-1], 1) if df["res_fn_hz"].notna().any() else None
        for a in ("x", "y", "z"):
            c = f"disp_{a}_mm"
            if c in df.columns:
                st[f"disp_{a}_max_mm"] = _f(df[c].max(), 3)
                st[f"disp_{a}_mean_mm"] = _f(df[c].mean(), 3)
        if "pitch_static_deg" in df.columns and "speed_kmh" in df.columns:
            slow = df.loc[df["speed_kmh"] < 10, "pitch_static_deg"].median()
            fast = df.loc[df["speed_kmh"] > 70, "pitch_static_deg"].median()
            st["pitch_change_70kmh_deg"] = _f(fast - slow, 2) if pd.notna(slow) and pd.notna(fast) else None
        out["structure"] = st

    if "load_mean" in df.columns:
        out["load"] = _stats(df["load_mean"], 3)

    if "batt_v" in df.columns and df["batt_v"].notna().any():
        out["power"] = {
            "batt_v_start": _f(df["batt_v"].dropna().iloc[0]), "batt_v_end": _f(df["batt_v"].dropna().iloc[-1]),
            "batt_a_mean": _f(df["batt_a"].mean()), "energy_wh": _f(df["energy_wh"].iloc[-1], 1)
            if "energy_wh" in df.columns else None,
            "soc_min": _f(df["batt_soc"].min(), 1) if "batt_soc" in df.columns else None,
        }
    if "pi_undervolt" in df.columns:
        out.setdefault("power", {})["undervolt_s"] = int((df["pi_undervolt"] == 1).sum())

    q = {}
    for c in ("missed", "imu_err"):
        if c in df.columns:
            q[f"{c}_total"] = int(df[c].fillna(0).sum())
            q[f"{c}_windows"] = int((df[c] > 0).sum())
    if "jitter_max_us" in df.columns:
        q["jitter_max_us"] = _f(df["jitter_max_us"].max(), 0)
    if "teensy_dropped" in df.columns:
        q["teensy_dropped"] = _f(df["teensy_dropped"].max(), 0)
    if q:
        out["quality"] = q
    return out


def vibration_by_speed(df: pd.DataFrame, step_kmh: int = 20) -> pd.DataFrame:
    """Mean vibration per speed band: shows how road/engine vibration grows with speed."""
    cols = [c for c in ("ax_rms", "ay_rms", "az_rms", "vib_rms_total") if c in df.columns]
    if "speed_kmh" not in df.columns or not cols:
        return pd.DataFrame()
    d = df[["speed_kmh"] + cols].dropna(subset=["speed_kmh"])
    if d.empty:
        return pd.DataFrame()
    top = int(np.ceil(max(d["speed_kmh"].max(), step_kmh) / step_kmh) * step_kmh)
    bins = list(range(0, top + step_kmh, step_kmh))
    d = d.assign(band=pd.cut(d["speed_kmh"], bins=bins, right=False,
                             labels=[f"{b}-{b + step_kmh}" for b in bins[:-1]]))
    t = d.groupby("band", observed=True)[cols].mean().round(3)
    t.insert(0, "seconds", d.groupby("band", observed=True).size())
    t.columns = ["seconds"] + [label(c) for c in cols]
    return t.reset_index().rename(columns={"band": "speed band (km/h)"})


def flatten(summary: dict) -> pd.DataFrame:
    """Summary dict -> two-column table (section.key, value) for display / export."""
    rows = []

    def walk(prefix, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(f"{prefix}.{k}" if prefix else k, v)
        else:
            rows.append({"item": prefix, "value": "--" if obj is None else str(obj)})   # one dtype for display
    walk("", summary)
    return pd.DataFrame(rows)
