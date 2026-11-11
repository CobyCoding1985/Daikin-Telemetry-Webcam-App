#!/usr/bin/env python3
"""
report.py - Command line: clean a session and export everything for the presentation.

  python report.py logs/session_20261110_160000.csv
  python report.py logs/session_*.csv --out reportes        (several sessions)
  python report.py session.csv --cdn                         (smaller HTML, needs internet to view)

Writes, next to each CSV (or in --out):
  <name>_clean.csv      cleaned data + derived columns (open in Excel / pandas)
  <name>_summary.json   summary numbers
  <name>_report.html    all charts, interactive, one file (works offline unless --cdn)
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import os

import pandas as pd

from telemetry import plots
from telemetry.analysis import flatten, summarize, vibration_by_speed
from telemetry.cleaning import CleanOptions, CleanResult, load_and_clean, read_raw

CSS = """
body{font-family:system-ui,Segoe UI,Arial,sans-serif;background:#121212;color:#e0e0e0;margin:0;padding:16px}
.wrap{max-width:1150px;margin:0 auto}
h1{margin:4px 0 2px} h2{color:#bb86fc;font-size:1.15em;margin:26px 0 8px}
.sub{color:#9e9e9e;font-size:.9em}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(175px,1fr));gap:10px;margin:14px 0}
.tile{background:#1e1e1e;border:1px solid #333;border-radius:12px;padding:10px 14px}
.tile span{display:block;color:#9e9e9e;font-size:.8em}.tile b{font-size:1.35em;color:#00bcd4}
.card{background:#1e1e1e;border:1px solid #333;border-radius:12px;padding:8px;margin:10px 0}
table{border-collapse:collapse;width:100%;font-size:.85em;font-variant-numeric:tabular-nums}
th,td{padding:4px 8px;border-bottom:1px solid #2c2c2c;text-align:left}th{color:#9e9e9e;font-weight:normal}
details summary{cursor:pointer;color:#9e9e9e}
"""


def tiles(summary: dict) -> list[tuple[str, str]]:
    s, d = summary.get("session", {}), summary.get("driving", {})
    v, q = summary.get("vibration", {}), summary.get("quality", {})
    env, pw = summary.get("environment", {}), summary.get("power", {})

    def fmt(x, unit=""):
        return "--" if x is None else f"{x}{unit}"
    out = [
        ("Duration", fmt(s.get("duration_min"), " min")),
        ("Data completeness", fmt(s.get("completeness_pct"), " %")),
        ("Lost packets", fmt(s.get("lost_packets"))),
        ("Distance", fmt(d.get("distance_km"), " km")),
        ("Max speed", fmt(d.get("max_speed_kmh"), " km/h")),
        ("Time at >= 80 km/h", fmt(d.get("time_above_80_min"), " min")),
        ("Max accel / brake", f"{fmt(d.get('max_accel_ms2'))} / {fmt(d.get('max_brake_ms2'))} m/s^2"),
        ("Vertical vib. RMS (mean)", fmt(v.get("az", {}).get("rms", {}).get("mean"), " m/s^2")),
        ("Vib. vs speed (r)", fmt(v.get("corr_rms_total_vs_speed"))),
        ("Air temp (mean)", fmt(env.get("bme_temp", {}).get("mean"), " C")),
        ("Missed IMU samples", fmt(q.get("missed_total"))),
        ("Battery energy used", fmt(pw.get("energy_wh"), " Wh")),
    ]
    return out


def table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "<p class='sub'>none</p>"
    return df.to_html(index=False, border=0, escape=True, na_rep="--", float_format=lambda x: f"{x:.3f}")


def figures(res: CleanResult, raw: pd.DataFrame | None, tiles_map: bool = True) -> list[tuple[str, object]]:
    df, gaps = res.df, res.gaps
    return [
        ("Route", plots.fig_route(df, tiles=tiles_map)),
        ("Speed and acceleration", plots.fig_speed(df, gaps)),
        ("G-G diagram", plots.fig_gg(df)),
        ("Vibration over time", plots.fig_vibration(df, gaps)),
        ("Vibration vs speed", plots.fig_vib_vs_speed(df)),
        ("Dominant frequency vs speed", plots.fig_freq_vs_speed(df)),
        ("Spectrogram", plots.fig_spectrogram(raw) if raw is not None else None),
        ("Load cell", plots.fig_load(df, gaps)),
        ("Environment", plots.fig_lines(df, ["bme_temp", "bme_hum", "bme_press", "air_density"], "Environment (BME280)", gaps)),
        ("Power", plots.fig_lines(df, ["batt_v", "batt_a", "batt_soc"], "Battery", gaps)),
        ("Data quality", plots.fig_quality(df, gaps)),
        ("Correlations", plots.fig_correlation(df)),
    ]


def build_report_html(res: CleanResult, summary: dict, raw: pd.DataFrame | None = None,
                      offline: bool = True, tiles_map: bool = True) -> str:
    s = summary.get("session", {})
    parts = [f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>Telemetry report {html.escape(s.get('file', ''))}</title><style>{CSS}</style></head>"
             f"<body><div class='wrap'><h1>Suzuki Swift telemetry report</h1>"
             f"<div class='sub'>{html.escape(str(s.get('file', '')))} &middot; {html.escape(str(s.get('start', '')))}"
             f" to {html.escape(str(s.get('end', '')))} &middot; time: {html.escape(str(s.get('time_source', '')))}</div>"]
    parts.append("<div class='tiles'>" + "".join(
        f"<div class='tile'><span>{html.escape(k)}</span><b>{html.escape(v)}</b></div>" for k, v in tiles(summary)) + "</div>")
    first = True
    for title, fig in figures(res, raw, tiles_map):
        if fig is None:
            continue
        js = ("inline" if offline else "cdn") if first else False
        parts.append(f"<h2>{html.escape(title)}</h2><div class='card'>"
                     + fig.to_html(full_html=False, include_plotlyjs=js, config={"displaylogo": False}) + "</div>")
        first = False
    parts.append("<h2>Vibration by speed band</h2><div class='card'>" + table(vibration_by_speed(res.df)) + "</div>")
    parts.append("<h2>What the cleaning changed</h2><div class='card'>" + table(res.log_frame()) + "</div>")
    gaps = res.gaps.copy()
    parts.append("<h2>Missing data (gaps)</h2><div class='card'>" + table(gaps) + "</div>")
    parts.append("<h2>All summary values</h2><div class='card'><details><summary>show</summary>"
                 + table(flatten(summary)) + "</details></div>")
    parts.append("</div></body></html>")
    return "".join(parts)


def export_clean_csv(res: CleanResult) -> bytes:
    df = res.df.copy()
    df.insert(0, "time", df.pop("time").dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3])
    for c in df.select_dtypes("number").columns:       # keep GPS precision (1e-7 deg = 1 cm)
        df[c] = df[c].round(7 if c in ("lat", "lon") else 5)
    return df.to_csv(index=False).encode("ascii", "replace")


def raw_path_for(csv_path: str) -> str | None:
    p = csv_path[:-4] + "_raw.csv"
    return p if os.path.exists(p) else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean telemetry sessions and build an HTML report")
    ap.add_argument("csv", nargs="+", help="session CSV file(s), wildcards allowed")
    ap.add_argument("--out", help="output folder (default: next to each CSV)")
    ap.add_argument("--cdn", action="store_true", help="smaller HTML that loads plotly from the internet")
    ap.add_argument("--no-tiles", action="store_true", help="draw the route without a street map")
    a = ap.parse_args()

    paths = []
    for pat in a.csv:
        paths += glob.glob(pat) or [pat]
    paths = [p for p in paths if not p.endswith("_raw.csv") and not p.endswith("_clean.csv")]
    opt = CleanOptions()
    for path in paths:
        name = os.path.basename(path)
        res = load_and_clean(path, opt, name)
        summ = summarize(res, opt)
        rp = raw_path_for(path)
        raw = read_raw(rp) if rp else None
        out_dir = a.out or os.path.dirname(os.path.abspath(path))
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.join(out_dir, name[:-4])
        with open(base + "_clean.csv", "wb") as fh:
            fh.write(export_clean_csv(res))
        with open(base + "_summary.json", "w", encoding="ascii") as fh:
            json.dump(summ, fh, indent=2, default=str)
        with open(base + "_report.html", "w", encoding="utf-8") as fh:
            fh.write(build_report_html(res, summ, raw, offline=not a.cdn, tiles_map=not a.no_tiles))
        s = summ["session"]
        print(f"{name}: {s['duration_min']} min, {s['completeness_pct']} % complete, "
              f"{s['lost_packets']} lost packets -> {base}_report.html")


if __name__ == "__main__":
    main()
