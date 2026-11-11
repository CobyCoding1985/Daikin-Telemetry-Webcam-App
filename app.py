# Open Telemetry Analysis app in a browser:
# 1. cd telemetry_app
# 2. python -m streamlit run app.py
# 3. Select Network Url


from __future__ import annotations

import glob
import inspect
import json
import os
from dataclasses import asdict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from report import build_report_html, export_clean_csv
from telemetry import plots
from telemetry.analysis import flatten, summarize, vibration_by_speed
from telemetry.cleaning import CleanOptions, CleanResult, clean_session, read_csv, read_raw
from telemetry.schema import COLUMNS, label

st.set_page_config(page_title="Swift Telemetry Analysis", page_icon=":racing_car:", layout="wide")

DEFAULT_FOLDERS = ["/media/cobyraspberry11/USB DISK/logs", os.path.expanduser("~/Documents/logs"), "demo"]
PLOT_CFG = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}}


# ------------------------------------------------------------------------------- caching
@st.cache_data(show_spinner="Cleaning data...")
def process(data: bytes, name: str, opt_dict: dict) -> tuple[CleanResult, dict]:
    opt = CleanOptions(**opt_dict)
    res = clean_session(read_csv(data), opt, name)
    return res, summarize(res, opt)


@st.cache_data(show_spinner="Reading raw file...")
def process_raw(data: bytes) -> pd.DataFrame | None:
    return read_raw(data)


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def _width_kw(fn) -> dict:
    """New Streamlit: width="stretch". Older: use_container_width=True (width was a number or
    went to the deprecated Plotly **kwargs). Decided from the installed function signature."""
    try:
        p = inspect.signature(fn).parameters.get("width")
    except (TypeError, ValueError):
        p = None
    if p is not None and isinstance(p.default, str):
        return {"width": "stretch"}
    return {"use_container_width": True}


def full_width(fn, *args, **kw):
    return fn(*args, **_width_kw(fn), **kw)


def show(fig: go.Figure) -> None:
    full_width(st.plotly_chart, fig, config=PLOT_CFG)


def table(df: pd.DataFrame, **kw) -> None:
    full_width(st.dataframe, df, hide_index=True, **kw)


# ------------------------------------------------------------------------------- sidebar: data
st.sidebar.title("Swift Telemetry")
# Online (Streamlit Cloud) there are no local folders: start in Upload mode
has_folder = any(os.path.isdir(f) for f in DEFAULT_FOLDERS[:-1])
mode = st.sidebar.radio("Data source", ["Folder", "Upload"], index=0 if has_folder else 1, horizontal=True)
sessions: dict[str, bytes] = {}
raw_bytes: dict[str, bytes] = {}

if mode == "Folder":
    start = next((f for f in DEFAULT_FOLDERS if os.path.isdir(f)), DEFAULT_FOLDERS[-1])
    folder = st.sidebar.text_input("Folder with session_*.csv", start)
    files = sorted(glob.glob(os.path.join(folder, "session_*.csv")), key=os.path.getmtime, reverse=True)
    files = [f for f in files if not f.endswith(("_raw.csv", "_clean.csv"))]
    if not files:
        st.sidebar.warning("No session_*.csv in that folder")
    for f in files:
        sessions[os.path.basename(f)] = f            # path, read lazily
        rp = f[:-4] + "_raw.csv"
        if os.path.exists(rp):
            raw_bytes[os.path.basename(f)] = rp
else:
    ups = st.sidebar.file_uploader("Session CSV (one or more)", type="csv", accept_multiple_files=True)
    for u in ups or []:
        if u.name.endswith("_raw.csv"):
            raw_bytes[u.name[:-8] + ".csv"] = u.getvalue()
        else:
            sessions[u.name] = u.getvalue()

if not sessions:
    st.title("Swift Telemetry Analysis")
    st.info("Choose a folder with session CSV files or upload one in the sidebar.  \n"
            "No data yet? Run `python make_demo_data.py --raw` and use the folder `demo`.")
    st.stop()

name = st.sidebar.selectbox("Session", list(sessions))


def get_bytes(src) -> bytes:
    return read_bytes(src) if isinstance(src, str) else src


# ------------------------------------------------------------------------------- sidebar: options
with st.sidebar.expander("Cleaning thresholds"):
    d = CleanOptions()
    opt = CleanOptions(
        timezone=st.text_input("Time zone", d.timezone),
        min_sats=st.number_input("Min satellites", 0, 20, d.min_sats),
        max_hdop=st.number_input("Max HDOP", 0.5, 99.0, d.max_hdop, 0.5),
        max_gps_speed_kmh=st.number_input("Max position-jump speed (km/h)", 50.0, 1000.0, d.max_gps_speed_kmh, 10.0),
        max_long_accel_ms2=st.number_input("Max speed change (m/s^2)", 2.0, 50.0, d.max_long_accel_ms2, 1.0),
        min_vib_amp_ms2=st.number_input("Min amplitude for dominant freq (m/s^2)", 0.0, 2.0, d.min_vib_amp_ms2, 0.01),
        gap_s=st.number_input("Gap if step longer than (s)", 1.1, 30.0, d.gap_s, 0.5),
        moving_kmh=st.number_input("Moving above (km/h)", 0.0, 30.0, d.moving_kmh, 1.0),
    )

res, summary = process(get_bytes(sessions[name]), name, asdict(opt))
raw = process_raw(get_bytes(raw_bytes[name])) if name in raw_bytes else None
df_all = res.df

with st.sidebar.expander("View", expanded=True):
    smooth_s = st.slider("Smoothing (s)", 1, 30, 1, help="moving average for line charts, 1 = off")
    tiles = st.checkbox("Street map behind route (needs internet)", True)
    tmax = float(df_all["elapsed_min"].max()) if len(df_all) else 1.0
    t0, t1 = st.slider("Time window (min from start)", 0.0, max(tmax, 0.1), (0.0, max(tmax, 0.1)), 0.1)
    moving_only = st.checkbox("Only while moving (vibration charts)", False)

df = df_all[(df_all["elapsed_min"] >= t0) & (df_all["elapsed_min"] <= t1)]
gaps = res.gaps
if len(df) and not gaps.empty:
    gaps = gaps[(gaps["end"] >= df["time"].iloc[0]) & (gaps["start"] <= df["time"].iloc[-1])]
df_vib = df[df["speed_kmh"] > opt.moving_kmh] if moving_only and "speed_kmh" in df.columns else df

# ------------------------------------------------------------------------------- header
s = summary["session"]
st.title("Swift Telemetry Analysis")
st.caption(f"{name}  |  {s['start']} to {s['end']}  |  time: {s['time_source']}  |  format {s['format']}")

tabs = st.tabs(["Summary", "Route & speed", "Vibration", "Load cell", "Environment", "Power",
                "Data quality", "Spectrogram", "Explore", "Compare", "Data & export"])

# ---- Summary
with tabs[0]:
    dr, vb = summary.get("driving", {}), summary.get("vibration", {})
    q, pw = summary.get("quality", {}), summary.get("power", {})
    def val(x):
        return "--" if x is None else x
    c = st.columns(6)
    c[0].metric("Duration (min)", val(s["duration_min"]))
    c[1].metric("Completeness (%)", val(s["completeness_pct"]), help="rows received / seconds in the session")
    c[2].metric("Lost packets", val(s["lost_packets"]))
    c[3].metric("Distance (km)", val(dr.get("distance_km")))
    c[4].metric("Max speed (km/h)", val(dr.get("max_speed_kmh")))
    c[5].metric("At >= 80 km/h (min)", val(dr.get("time_above_80_min")))
    c = st.columns(6)
    c[0].metric("Max accel (m/s^2)", val(dr.get("max_accel_ms2")))
    c[1].metric("Max brake (m/s^2)", val(dr.get("max_brake_ms2")))
    c[2].metric("Vertical vib. RMS (m/s^2)", val(vb.get("az", {}).get("rms", {}).get("mean")))
    c[3].metric("Vibration vs speed (r)", val(vb.get("corr_rms_total_vs_speed")))
    c[4].metric("Missed IMU samples", val(q.get("missed_total")))
    c[5].metric("Battery energy (Wh)", val(pw.get("energy_wh")))
    left, right = st.columns([3, 2])
    with left:
        st.subheader("What the cleaning changed")
        table(res.log_frame())
    with right:
        st.subheader("Missing data")
        table(res.gaps)
        st.subheader("Vibration by speed band")
        table(vibration_by_speed(df_vib))
    with st.expander("All summary values"):
        table(flatten(summary))

# ---- Route & speed
with tabs[1]:
    color_opts = [c for c in ("speed_kmh", "vib_rms_total", "az_rms", "long_accel_ms2", "alt_m", "load_mean",
                              "bme_temp") if c in df.columns]
    color = st.selectbox("Colour route by", color_opts, format_func=label) if color_opts else "speed_kmh"
    show(plots.fig_route(df, color=color, tiles=tiles))
    show(plots.fig_speed(df, gaps, smooth_s))
    a, b = st.columns(2)
    with a:
        show(plots.fig_gg(df))
    with b:
        show(plots.fig_histogram(df, "speed_kmh"))

# ---- Vibration
with tabs[2]:
    show(plots.fig_vibration(df_vib, gaps, smooth_s))
    a, b = st.columns(2)
    with a:
        show(plots.fig_vib_vs_speed(df_vib))
    with b:
        ax = st.radio("Axis", ["az", "ax", "ay"], horizontal=True, format_func=lambda x: x[1].upper())
        show(plots.fig_freq_vs_speed(df_vib, ax))

# ---- Load cell
with tabs[3]:
    show(plots.fig_load(df, gaps, smooth_s))
    show(plots.fig_histogram(df, "load_mean"))

# ---- Environment
with tabs[4]:
    show(plots.fig_lines(df, ["bme_temp", "bme_hum", "bme_press", "air_density", "alt_m"],
                         "Environment (BME280) and altitude", gaps, smooth_s))

# ---- Power
with tabs[5]:
    show(plots.fig_lines(df, ["batt_v", "batt_a", "batt_w", "batt_soc", "energy_wh"], "Battery", gaps, smooth_s))
    if "pi_undervolt" in df.columns and (df["pi_undervolt"] == 1).any():
        st.warning(f"Raspberry Pi under-voltage during {int((df['pi_undervolt'] == 1).sum())} s")

# ---- Data quality
with tabs[6]:
    show(plots.fig_quality(df, gaps))

# ---- Spectrogram
with tabs[7]:
    if raw is None:
        st.info("This session has no raw file. Set SAVE_RAW = True in telemetria_final.py to record "
                "the 512 Hz accelerometer signal (about 50 MB per hour).")
    else:
        a, b = st.columns(2)
        axis = a.radio("Axis", ["az", "ax", "ay"], horizontal=True, key="spec_axis",
                       format_func=lambda x: x[1].upper())
        fmax = b.slider("Max frequency (Hz)", 20, 256, 128)
        show(plots.fig_spectrogram(raw, axis, fmax=fmax))

# ---- Explore: any variable against any other
with tabs[8]:
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()]
    a, b, c3 = st.columns(3)
    xcol = a.selectbox("X", ["time"] + num, format_func=lambda c: "time" if c == "time" else label(c))
    ycol = b.selectbox("Y", num, index=num.index("speed_kmh") if "speed_kmh" in num else 0, format_func=label)
    ccol = c3.selectbox("Colour", ["(none)"] + num, format_func=lambda c: c if c == "(none)" else label(c))
    dd = df.dropna(subset=[ycol])
    mk = dict(size=5, color=dd[ccol], colorscale="Turbo", showscale=True) if ccol != "(none)" else dict(size=5, color=plots.CYAN)
    fig = go.Figure(go.Scattergl(x=dd[xcol], y=dd[ycol], mode="markers" if xcol != "time" else "lines+markers", marker=mk))
    fig.update_xaxes(title_text="time" if xcol == "time" else label(xcol))
    fig.update_yaxes(title_text=label(ycol))
    show(plots._style(fig, f"{label(ycol)} vs {'time' if xcol == 'time' else label(xcol)}", 480))
    show(plots.fig_correlation(df))

# ---- Compare sessions
with tabs[9]:
    pick = st.multiselect("Sessions to compare", list(sessions), default=[name])
    var = st.selectbox("Variable", [c for c in COLUMNS if c in df_all.columns], format_func=label,
                       index=[c for c in COLUMNS if c in df_all.columns].index("speed_kmh") if "speed_kmh" in df_all.columns else 0)
    fig = go.Figure()
    rows = []
    for nm in pick:
        r, sm = process(get_bytes(sessions[nm]), nm, asdict(opt))
        if var in r.df.columns:
            fig.add_trace(go.Scatter(x=r.df["elapsed_min"], y=plots.smooth(r.df[var], smooth_s), name=nm, mode="lines"))
        rows.append({"session": nm, "minutes": sm["session"]["duration_min"],
                     "complete %": sm["session"]["completeness_pct"],
                     "km": sm.get("driving", {}).get("distance_km"),
                     "max km/h": sm.get("driving", {}).get("max_speed_kmh"),
                     "vib Z rms": sm.get("vibration", {}).get("az", {}).get("rms", {}).get("mean"),
                     "lost": sm["session"]["lost_packets"]})
    fig.update_xaxes(title_text="minutes from start")
    fig.update_yaxes(title_text=label(var))
    show(plots._style(fig, f"{label(var)} by session", 460))
    table(pd.DataFrame(rows))

# ---- Data & export
with tabs[10]:
    full_width(st.dataframe, df, height=380)
    base = name[:-4]
    a, b, c3 = st.columns(3)
    a.download_button("Cleaned CSV", export_clean_csv(res), f"{base}_clean.csv", "text/csv")
    b.download_button("Summary JSON", json.dumps(summary, indent=2, default=str), f"{base}_summary.json",
                      "application/json")
    if c3.button("Build HTML report"):
        st.session_state["report"] = (name, build_report_html(res, summary, raw, offline=True, tiles_map=tiles))
    rep = st.session_state.get("report")
    if rep and rep[0] == name:
        c3.download_button("Download HTML report", rep[1], f"{base}_report.html", "text/html")