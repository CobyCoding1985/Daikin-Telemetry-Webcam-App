"""
app.py - Interactive analysis of the telemetry sessions (Streamlit).

  streamlit run app.py

Opens in the browser (http://localhost:8501). Pick a session from a folder or upload a CSV,
adjust the cleaning thresholds in the sidebar, and every chart / table updates.
"""

from __future__ import annotations

import glob
import inspect
import json
import os
import time
from dataclasses import asdict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from report import build_report_html, export_clean_csv
from telemetry import plots
from telemetry.analysis import flatten, summarize, vibration_by_speed
from telemetry.cleaning import CleanOptions, CleanResult, clean_session, read_csv, read_raw
from telemetry.schema import COLUMNS, label
from telemetry.video import FrameReader, align, export_mp4, hud, nearest, read_frames_index, row_at

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


@st.cache_data(show_spinner="Reading video index...")
def process_frames(data: bytes) -> pd.DataFrame:
    return read_frames_index(data)


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


_chart_no = 0


def show(fig: go.Figure) -> None:
    """Every chart gets its own key (the script re-runs top to bottom, so keys stay stable):
    two identical figures in different tabs would otherwise collide."""
    global _chart_no
    _chart_no += 1
    full_width(st.plotly_chart, fig, config=PLOT_CFG, key=f"chart_{_chart_no}")


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
    files = [f for f in files if not f.endswith(("_raw.csv", "_clean.csv", "_frames.csv"))]
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
        elif u.name.endswith("_frames.csv"):
            continue                                 # video index: upload it in the Video tab
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

def _present(*cols) -> bool:
    return any(c in df_all.columns and df_all[c].notna().any() for c in cols)


TAB_NAMES = ["Summary", "Route & speed", "Video", "Vibration", "Airspeed", "Structure", "Environment"]
if _present("load_mean"):                    # sessions recorded before firmware 3.0
    TAB_NAMES.append("Load cell")
if _present("batt_v"):
    TAB_NAMES.append("Battery")
TAB_NAMES += ["Data quality", "Spectrogram", "Explore", "Compare", "Data & export"]
TABS = dict(zip(TAB_NAMES, st.tabs(TAB_NAMES)))
tab_sum, tab_route, tab_video, tab_vib = TABS["Summary"], TABS["Route & speed"], TABS["Video"], TABS["Vibration"]
tab_air, tab_struct, tab_env = TABS["Airspeed"], TABS["Structure"], TABS["Environment"]
tab_load, tab_pow = TABS.get("Load cell"), TABS.get("Battery")
tab_q, tab_spec, tab_exp, tab_cmp, tab_data = (TABS["Data quality"], TABS["Spectrogram"], TABS["Explore"],
                                               TABS["Compare"], TABS["Data & export"])

# ---- Summary
with tab_sum:
    dr, vb = summary.get("driving", {}), summary.get("vibration", {})
    q, sr, ai = summary.get("quality", {}), summary.get("structure", {}), summary.get("airspeed", {})
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
    c[5].metric("Resonance WARNING (s)", val(sr.get("warning_s")))
    c = st.columns(6)
    c[0].metric("Max airspeed (km/h)", val(ai.get("max_airspeed_kmh")))
    c[1].metric("Mean headwind >40 km/h", val(ai.get("mean_headwind_kmh_above_40")))
    c[2].metric("Pitot vs GPS q (r)", val(ai.get("corr_dp_vs_q_gps")))
    c[3].metric("Max displacement Z (mm)", val(sr.get("disp_z_max_mm")))
    c[4].metric("Resonance caution (s)", val(sr.get("caution_s")))
    c[5].metric("Natural freq. (Hz)", val(sr.get("natural_freq_hz")))
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
with tab_route:
    color_opts = [c for c in ("speed_kmh", "airspeed_kmh", "wind_kmh", "vib_rms_total", "disp_total_mm", "res_level",
                              "az_rms", "long_accel_ms2", "alt_m", "bme_temp") if c in df.columns]
    color = st.selectbox("Colour route by", color_opts, format_func=label) if color_opts else "speed_kmh"
    show(plots.fig_route(df, color=color, tiles=tiles))
    show(plots.fig_speed(df, gaps, smooth_s))
    a, b = st.columns(2)
    with a:
        show(plots.fig_gg(df))
    with b:
        show(plots.fig_histogram(df, "speed_kmh"))

# ---- Video (synchronised with the data)
def hud_lines(r, t: pd.Timestamp) -> list[str]:
    if r is None:
        return [f"{t:%H:%M:%S}  (no telemetry at this moment)"]
    def g(col, fmt):
        v = r.get(col)
        return "--" if v is None or pd.isna(v) else format(v, fmt)
    return [f"{t:%H:%M:%S}   {g('speed_kmh', '.0f')} km/h   accel {g('long_accel_ms2', '+.2f')} m/s2",
            f"air {g('airspeed_kmh', '.0f')} km/h   vib Z {g('az_rms', '.2f')} m/s2 @ {g('az_freq', '.0f')} Hz"
            + ("   RESONANCE" if (r.get("res_level") or 0) >= 2 else "")]


def shift_video(dt: float, lo: float, hi: float) -> None:
    st.session_state["vid_t"] = float(min(max(st.session_state.get("vid_t", lo) + dt, lo), hi))


with tab_video:
    vid_src = idx_src = None
    if mode == "Folder":
        vb = sessions[name][:-4]
        if os.path.exists(vb + "_video.mjpeg") and os.path.exists(vb + "_frames.csv"):
            vid_src, idx_src = vb + "_video.mjpeg", vb + "_frames.csv"
    else:
        vups = st.file_uploader("Video of this session: _video.mjpeg and _frames.csv", type=["mjpeg", "mjpg", "csv"],
                                accept_multiple_files=True, key="vid_up")
        for u in vups or []:
            if u.name.endswith("_frames.csv"):
                idx_src = u.getvalue()
            elif u.name.lower().endswith((".mjpeg", ".mjpg")):
                vid_src = u.getvalue()
    if vid_src is None or idx_src is None:
        st.info("No video for this session. The Pi saves it with every recording when VIDEO_RECORD = True "
                "(telemetria_final.py v2.3): session_X_video.mjpeg + session_X_frames.csv next to the CSV.")
    else:
        frames = align(process_frames(get_bytes(idx_src)), df_all)
        reader = FrameReader(vid_src)
        if len(frames) < 2:
            st.warning("The video has fewer than 2 frames")
        else:
            lo, hi = float(max(frames["elapsed_s"].min(), 0.0)), float(frames["elapsed_s"].max())
            fps = (len(frames) - 1) / max(1e-6, (frames["time"].iloc[-1] - frames["time"].iloc[0]).total_seconds())
            if "vid_t_next" in st.session_state:                 # set by Play before the slider exists
                st.session_state["vid_t"] = st.session_state.pop("vid_t_next")
            st.session_state["vid_t"] = float(min(max(st.session_state.get("vid_t", lo), lo), hi))
            b = st.columns(6)
            b[0].button("-10 s", on_click=shift_video, args=(-10, lo, hi), key="vb1")
            b[1].button("-1 s", on_click=shift_video, args=(-1, lo, hi), key="vb2")
            b[2].button("-1 frame", on_click=shift_video, args=(-1 / fps, lo, hi), key="vb3")
            b[3].button("+1 frame", on_click=shift_video, args=(1 / fps, lo, hi), key="vb4")
            b[4].button("+1 s", on_click=shift_video, args=(1, lo, hi), key="vb5")
            b[5].button("+10 s", on_click=shift_video, args=(10, lo, hi), key="vb6")
            t_sel = st.slider("Time (seconds from start of the data)", lo, hi, key="vid_t", step=0.1, format="%.1f s")
            k = nearest(frames, t_sel)
            fr = frames.iloc[k]
            jpeg = reader.get(fr["offset"], fr["size"])
            r = row_at(df_all, fr["time"])

            left, right = st.columns([3, 2])
            with left:
                show_hud = st.checkbox("Show data on the image", True, key="vid_hud")
                pic = st.empty()
                img = hud(jpeg, hud_lines(r, fr["time"])) if show_hud else jpeg
                full_width(pic.image, img)
                st.download_button("Download this frame (PNG)", hud(jpeg, hud_lines(r, fr["time"])),
                                   f"{name[:-4]}_frame_{int(fr['frame'])}.png", "image/png")
            with right:
                st.caption(f"frame {int(fr['frame'])} of {len(frames)}  |  {fps:.1f} fps  |  {fr['time']:%H:%M:%S.%f}"[:-3])
                m = st.columns(2)
                def mv(col, fmt):
                    return "--" if r is None or col not in r or pd.isna(r[col]) else format(r[col], fmt)
                m[0].metric("Speed (km/h)", mv("speed_kmh", ".1f"))
                m[1].metric("Accel (m/s^2)", mv("long_accel_ms2", "+.2f"))
                m[0].metric("Vib. RMS Z (m/s^2)", mv("az_rms", ".3f"))
                m[1].metric("Dominant freq Z (Hz)", mv("az_freq", ".1f"))
                m[0].metric("Airspeed (km/h)", mv("airspeed_kmh", ".1f"))
                m[1].metric("Altitude (m)", mv("alt_m", ".0f"))
                p = st.columns(3)
                dur = p[0].selectbox("Play", [5, 10, 30, 60], index=1, format_func=lambda x: f"{x} s", key="vid_dur")
                spd = p[1].selectbox("Speed", [0.5, 1.0, 2.0, 4.0], index=1, format_func=lambda x: f"{x:g}x", key="vid_spd")
                play = p[2].button("Play", key="vid_play")

            ctx = plots.fig_speed(df_all, res.gaps, smooth_s)
            ctx.add_vline(x=fr["time"], line=dict(color=plots.RED, width=2))
            show(ctx)

            with st.expander("Export MP4 (plays on any phone / PowerPoint)"):
                e = st.columns([3, 1])
                r0, r1 = e[0].slider("Range (s)", lo, hi, (lo, hi), step=1.0, key="mp4_rng")
                burn = e[1].checkbox("Burn data on video", True, key="mp4_burn")
                sub = frames[(frames["elapsed_s"] >= r0) & (frames["elapsed_s"] <= r1)]
                st.caption(f"{len(sub)} frames, {(r1 - r0) / 60:.1f} min. About 1-3 min of processing per "
                           "hour of video with data burned in, much less without.")
                if st.button("Create MP4", key="mp4_go") and len(sub) > 1:
                    import tempfile
                    bar = st.progress(0.0, text="Encoding...")
                    # Folder mode: keep the MP4 next to the session CSV; Upload mode: temporary file
                    out_dir = os.path.dirname(sessions[name]) if mode == "Folder" else tempfile.gettempdir()
                    out = os.path.join(out_dir, f"{name[:-4]}_{int(r0)}-{int(r1)}s.mp4")
                    try:
                        export_mp4(reader, sub, out, fps,
                                   overlay=(lambda f: hud_lines(row_at(df_all, f["time"]), f["time"])) if burn else None,
                                   progress=lambda x: bar.progress(min(1.0, x), text=f"Encoding... {x:.0%}"))
                        with open(out, "rb") as fh:
                            st.session_state["mp4"] = (name, os.path.basename(out), fh.read())
                        if mode == "Folder":
                            st.success(f"Saved: {out}")
                        else:
                            os.remove(out)
                        bar.progress(1.0, text="Done")
                    except Exception as ex:                       # missing ffmpeg, disk full...
                        hint = "  Install it with: python -m pip install imageio-ffmpeg" \
                            if isinstance(ex, ImportError) else ""
                        st.error(f"Could not create the MP4: {ex}.{hint}")
                mp4 = st.session_state.get("mp4")
                if mp4 and mp4[0] == name:
                    st.download_button(f"Download {mp4[1]} ({len(mp4[2]) / 1e6:.1f} MB)", mp4[2], mp4[1], "video/mp4")

            if play:
                t_end = min(hi, t_sel + dur)
                sub = frames[(frames["elapsed_s"] >= t_sel) & (frames["elapsed_s"] <= t_end)]
                wall0 = time.monotonic()
                for _, f in sub.iterrows():
                    late = (time.monotonic() - wall0) - (f["elapsed_s"] - t_sel) / spd
                    if late > 0.25:
                        continue                                  # behind real time: skip frames
                    if late < 0:
                        time.sleep(-late)
                    jp = reader.get(f["offset"], f["size"])
                    img = hud(jp, hud_lines(row_at(df_all, f["time"]), f["time"])) if show_hud else jp
                    full_width(pic.image, img)
                st.session_state["vid_t_next"] = float(t_end)
                st.rerun()

# ---- Vibration
with tab_vib:
    show(plots.fig_vibration(df_vib, gaps, smooth_s))
    a, b = st.columns(2)
    with a:
        show(plots.fig_vib_vs_speed(df_vib))
    with b:
        ax = st.radio("Axis", ["az", "ax", "ay"], horizontal=True, format_func=lambda x: x[1].upper())
        show(plots.fig_freq_vs_speed(df_vib, ax))

# ---- Airspeed (pitot)
with tab_air:
    show(plots.fig_airspeed(df, gaps, smooth_s))
    a, b = st.columns(2)
    with a:
        show(plots.fig_dp_vs_q(df))
    with b:
        show(plots.fig_histogram(df, "wind_kmh"))
    st.caption("Points above the 1:1 line = headwind, below = tailwind. With no wind they should sit on it: "
               "if they are all off by the same factor, the tube position (boundary layer) or the zero is off.")

# ---- Structure (box support)
with tab_struct:
    show(plots.fig_structure(df, gaps, smooth_s))
    a, b = st.columns(2)
    with a:
        show(plots.fig_freq_vs_speed(df, "az"))
    with b:
        if "disp_total_mm" in df.columns and "speed_kmh" in df.columns:
            fig = go.Figure(go.Scatter(x=df["speed_kmh"], y=df["disp_total_mm"], mode="markers",
                                       marker=dict(size=5, color=df.get("res_level"), colorscale=[[0, plots.CYAN], [0.5, plots.AMBER], [1, plots.RED]],
                                                   cmin=0, cmax=2)))
            fig.update_xaxes(title_text="speed (km/h)")
            fig.update_yaxes(title_text="displacement RMS total (mm)")
            show(plots._style(fig, "Displacement vs speed (colour = resonance level)", 460))
    st.caption("Displacement = vibration of the box (acceleration integrated twice, 3-200 Hz). The static pitch "
               "with the car acceleration removed shows how much the support bends under aerodynamic load: compare "
               "it at low speed and at 70-80 km/h on a level road.")

# ---- Load cell (old sessions)
if tab_load is not None:
    with tab_load:
        show(plots.fig_load(df, gaps, smooth_s))
        show(plots.fig_histogram(df, "load_mean"))

# ---- Environment
with tab_env:
    show(plots.fig_lines(df, ["bme_temp", "bme_hum", "bme_press", "air_density", "alt_m"],
                         "Environment (BME280) and altitude", gaps, smooth_s))

# ---- Battery (old sessions)
if tab_pow is not None:
    with tab_pow:
        show(plots.fig_lines(df, ["batt_v", "batt_a", "batt_w", "batt_soc", "energy_wh"], "Battery", gaps, smooth_s))

# ---- Data quality
with tab_q:
    if "pi_undervolt" in df.columns and (df["pi_undervolt"] == 1).any():
        st.warning(f"Raspberry Pi under-voltage during {int((df['pi_undervolt'] == 1).sum())} s")
    show(plots.fig_quality(df, gaps))

# ---- Spectrogram
with tab_spec:
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
with tab_exp:
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
with tab_cmp:
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
with tab_data:
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
