"""
plots.py - One function per chart. Each takes the cleaned DataFrame and returns a plotly Figure,
so the same charts are used by the Streamlit app (app.py) and the HTML report (report.py).

To add a chart: write fig_something(df) here, then call it from app.py and from report.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .cleaning import with_gap_breaks
from .schema import AXES, label

# ---- look: same colours as the live web dashboard ----
BG, CARD, GRID, TEXT, MUTED = "#121212", "#1e1e1e", "#2c2c2c", "#e0e0e0", "#9e9e9e"
CYAN, PURPLE, GREEN, AMBER, RED = "#00bcd4", "#bb86fc", "#4caf50", "#ffb300", "#ef5350"
AXIS_COLOR = {"ax": CYAN, "ay": PURPLE, "az": AMBER}


def _style(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.01, font=dict(size=16, color=TEXT)),
        paper_bgcolor=CARD, plot_bgcolor=CARD, font=dict(color=TEXT, size=12),
        height=height, margin=dict(l=60, r=20, t=50, b=40),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right", bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig


def _x(df: pd.DataFrame):
    return df["time"]


def _has(df, *cols) -> bool:
    return all(c in df.columns and df[c].notna().any() for c in cols)


def empty(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(color=MUTED, size=14))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _style(fig, "", 200)


def smooth(s: pd.Series, window_s: int) -> pd.Series:
    return s if window_s <= 1 else s.rolling(window_s, center=True, min_periods=1).mean()


def shade_gaps(fig: go.Figure, gaps: pd.DataFrame, rows: int = 1) -> go.Figure:
    for _, g in gaps.iterrows():
        for r in range(1, rows + 1):
            fig.add_vrect(x0=g["start"], x1=g["end"], fillcolor=RED, opacity=0.15, line_width=0,
                          row=r if rows > 1 else None, col=1 if rows > 1 else None)
    return fig


# ================================ DRIVING ================================
def fig_speed(df: pd.DataFrame, gaps: pd.DataFrame | None = None, smooth_s: int = 1) -> go.Figure:
    if not _has(df, "speed_kmh"):
        return empty("No GPS speed in this session")
    d = with_gap_breaks(df)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4])
    fig.add_trace(go.Scatter(x=_x(d), y=smooth(d["speed_kmh"], smooth_s), name="GPS speed",
                             line=dict(color=GREEN, width=2)), 1, 1)
    fig.add_hline(y=80, line=dict(color=MUTED, dash="dot"), annotation_text="80 km/h", row=1, col=1)
    if "long_accel_ms2" in d.columns:
        fig.add_trace(go.Scatter(x=_x(d), y=smooth(d["long_accel_ms2"], smooth_s), name="longitudinal",
                                 line=dict(color=CYAN, width=1.5)), 2, 1)
    if "lat_accel_ms2" in d.columns:
        fig.add_trace(go.Scatter(x=_x(d), y=smooth(d["lat_accel_ms2"], smooth_s), name="lateral (est.)",
                                 line=dict(color=PURPLE, width=1)), 2, 1)
    fig.update_yaxes(title_text="km/h", row=1, col=1)
    fig.update_yaxes(title_text="m/s^2", row=2, col=1)
    if gaps is not None:
        shade_gaps(fig, gaps, rows=2)
    return _style(fig, "Speed and acceleration (red bands = missing data)", 520)


def fig_route(df: pd.DataFrame, color: str = "speed_kmh", tiles: bool = True) -> go.Figure:
    if not _has(df, "lat", "lon"):
        return empty("No valid GPS positions in this session")
    d = df.dropna(subset=["lat", "lon"])
    c = d[color] if color in d.columns else None
    hover = d["time"].dt.strftime("%H:%M:%S") + "<br>" + d.get("speed_kmh", pd.Series(index=d.index)).round(1).astype(str) + " km/h"
    marker = dict(size=6, color=c, colorscale="Turbo", showscale=c is not None,
                  colorbar=dict(title=label(color), thickness=12))
    if tiles and hasattr(go, "Scattermap"):
        fig = go.Figure(go.Scattermap(lat=d["lat"], lon=d["lon"], mode="markers", marker=marker,
                                      text=hover, hoverinfo="text"))
        fig.update_layout(map=dict(style="open-street-map", zoom=13,
                                   center=dict(lat=d["lat"].mean(), lon=d["lon"].mean())))
    else:                                       # offline: plain x/y with equal metres per degree
        fig = go.Figure(go.Scatter(x=d["lon"], y=d["lat"], mode="markers", marker=marker,
                                   text=hover, hoverinfo="text"))
        fig.update_yaxes(scaleanchor="x", scaleratio=1 / np.cos(np.radians(d["lat"].mean())), title_text="lat")
        fig.update_xaxes(title_text="lon")
    fig = _style(fig, f"Route coloured by {label(color)}", 560)
    fig.update_layout(hovermode="closest", margin=dict(l=10, r=10, t=50, b=10))
    return fig


def fig_gg(df: pd.DataFrame) -> go.Figure:
    """Friction circle: lateral vs longitudinal acceleration (classic motorsport chart)."""
    if not _has(df, "long_accel_ms2", "lat_accel_ms2"):
        return empty("Needs GPS speed and gyro Z")
    d = df.dropna(subset=["long_accel_ms2", "lat_accel_ms2"])
    fig = go.Figure(go.Scatter(x=d["lat_accel_ms2"] / 9.80665, y=d["long_accel_ms2"] / 9.80665, mode="markers",
                               marker=dict(size=5, color=d.get("speed_kmh"), colorscale="Turbo", showscale=True,
                                           colorbar=dict(title="km/h", thickness=12))))
    for g in (0.25, 0.5):
        th = np.linspace(0, 2 * np.pi, 100)
        fig.add_trace(go.Scatter(x=g * np.cos(th), y=g * np.sin(th), mode="lines", showlegend=False,
                                 line=dict(color=MUTED, dash="dot", width=1), hoverinfo="skip"))
    fig.update_xaxes(title_text="lateral (g)", range=[-0.6, 0.6])
    fig.update_yaxes(title_text="longitudinal (g)", range=[-0.6, 0.6], scaleanchor="x")
    fig = _style(fig, "G-G diagram (acceleration envelope)", 480)
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


# ================================ VIBRATION ================================
def fig_vibration(df: pd.DataFrame, gaps: pd.DataFrame | None = None, smooth_s: int = 1) -> go.Figure:
    if not _has(df, "az_rms"):
        return empty("No vibration data")
    d = with_gap_breaks(df)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("RMS (1 s window)", "Peak |a - mean|", "Dominant frequency"))
    for a in AXES:
        col = AXIS_COLOR[a]
        if f"{a}_rms" in d.columns:
            fig.add_trace(go.Scatter(x=_x(d), y=smooth(d[f"{a}_rms"], smooth_s), name=a[1].upper(),
                                     legendgroup=a, line=dict(color=col, width=1.5)), 1, 1)
        if f"{a}_peak" in d.columns:
            fig.add_trace(go.Scatter(x=_x(d), y=smooth(d[f"{a}_peak"], smooth_s), name=a[1].upper(),
                                     legendgroup=a, showlegend=False, line=dict(color=col, width=1)), 2, 1)
        if f"{a}_freq" in d.columns:
            fig.add_trace(go.Scatter(x=_x(d), y=d[f"{a}_freq"], name=a[1].upper(), legendgroup=a,
                                     showlegend=False, mode="markers", marker=dict(color=col, size=3)), 3, 1)
    fig.update_yaxes(title_text="m/s^2", row=1, col=1)
    fig.update_yaxes(title_text="m/s^2", row=2, col=1)
    fig.update_yaxes(title_text="Hz", row=3, col=1)
    if gaps is not None:
        shade_gaps(fig, gaps, rows=3)
    fig.update_annotations(font=dict(color=MUTED, size=12))
    return _style(fig, "Vibration (accelerometer)", 720)


def fig_vib_vs_speed(df: pd.DataFrame, step_kmh: int = 10) -> go.Figure:
    if not _has(df, "speed_kmh", "vib_rms_total"):
        return empty("Needs GPS speed and vibration")
    d = df.dropna(subset=["speed_kmh", "vib_rms_total"])
    fig = go.Figure()
    for a in AXES:
        if f"{a}_rms" in d.columns:
            fig.add_trace(go.Scatter(x=d["speed_kmh"], y=d[f"{a}_rms"], mode="markers", name=f"RMS {a[1].upper()}",
                                     marker=dict(color=AXIS_COLOR[a], size=4, opacity=0.45)))
    band = (d["speed_kmh"] // step_kmh) * step_kmh + step_kmh / 2
    m = d.groupby(band)["vib_rms_total"].mean()
    fig.add_trace(go.Scatter(x=m.index, y=m.values, mode="lines+markers", name="total (band mean)",
                             line=dict(color=GREEN, width=3)))
    fig.update_xaxes(title_text="speed (km/h)")
    fig.update_yaxes(title_text="vibration RMS (m/s^2)")
    fig = _style(fig, "Vibration vs speed", 460)
    fig.update_layout(hovermode="closest")
    return fig


def fig_freq_vs_speed(df: pd.DataFrame, axis: str = "az") -> go.Figure:
    """Engine orders show up as straight lines that jump at each gear change."""
    if not _has(df, "speed_kmh", f"{axis}_freq"):
        return empty("Needs GPS speed and dominant frequency")
    d = df.dropna(subset=["speed_kmh", f"{axis}_freq"])
    fig = go.Figure(go.Scatter(x=d["speed_kmh"], y=d[f"{axis}_freq"], mode="markers",
                               marker=dict(size=5, color=d[f"{axis}_amp"], colorscale="Plasma", showscale=True,
                                           colorbar=dict(title="amp m/s^2", thickness=12))))
    fig.update_xaxes(title_text="speed (km/h)")
    fig.update_yaxes(title_text=f"dominant frequency {axis[1].upper()} (Hz)")
    fig = _style(fig, f"Dominant vibration frequency vs speed ({axis[1].upper()} axis)", 460)
    fig.update_layout(hovermode="closest")
    return fig


def fig_spectrogram(raw: pd.DataFrame, axis: str = "az", win_s: float = 1.0, fmax: float = 256) -> go.Figure:
    """From session_*_raw.csv (512 Hz): FFT magnitude per window (time x frequency)."""
    if raw is None or raw.empty:
        return empty("No raw file (enable SAVE_RAW in telemetria_final.py to record one)")
    x = raw[axis].to_numpy()
    fs = 1.0 / np.median(np.diff(raw["t_s"].to_numpy()))
    n = int(round(win_s * fs))
    k = len(x) // n
    if k < 2:
        return empty("Raw file too short")
    seg = x[: k * n].reshape(k, n)
    seg = seg - seg.mean(axis=1, keepdims=True)
    w = np.hanning(n)
    spec = np.abs(np.fft.rfft(seg * w, axis=1)) * 2 / w.sum()
    f = np.fft.rfftfreq(n, 1 / fs)
    keep = f <= fmax
    z = 20 * np.log10(spec[:, keep].T + 1e-4)
    fig = go.Figure(go.Heatmap(x=np.arange(k) * win_s, y=f[keep], z=z, colorscale="Inferno",
                               colorbar=dict(title="dB re 1 m/s^2", thickness=12)))
    fig.update_xaxes(title_text="time (s)")
    fig.update_yaxes(title_text="frequency (Hz)")
    fig = _style(fig, f"Spectrogram, {axis[1].upper()} axis (raw 512 Hz)", 480)
    fig.update_layout(hovermode="closest")
    return fig


# ================================ OTHER SENSORS ================================
def fig_lines(df: pd.DataFrame, cols: list[str], title: str, gaps: pd.DataFrame | None = None,
              smooth_s: int = 1, colors=None) -> go.Figure:
    """One stacked panel per column (different units never share an axis)."""
    cols = [c for c in cols if _has(df, c)]
    if not cols:
        return empty(f"No data for: {title}")
    d = with_gap_breaks(df)
    colors = colors or [CYAN, PURPLE, AMBER, GREEN, RED]
    fig = make_subplots(rows=len(cols), cols=1, shared_xaxes=True, vertical_spacing=0.05)
    for i, c in enumerate(cols, start=1):
        fig.add_trace(go.Scatter(x=_x(d), y=smooth(d[c], smooth_s), name=label(c),
                                 line=dict(color=colors[(i - 1) % len(colors)], width=1.8)), i, 1)
        fig.update_yaxes(title_text=label(c), title_font=dict(size=11), row=i, col=1)
    if gaps is not None:
        shade_gaps(fig, gaps, rows=len(cols))
    return _style(fig, title, 160 + 150 * len(cols))


def fig_load(df: pd.DataFrame, gaps: pd.DataFrame | None = None, smooth_s: int = 1) -> go.Figure:
    if not _has(df, "load_mean"):
        return empty("No load cell data")
    d = with_gap_breaks(df)
    fig = go.Figure()
    if _has(d, "load_min", "load_max"):
        # one closed polygon per continuous stretch, so the band never bridges a gap
        band = d[["time", "load_min", "load_max"]]
        seg_id = band[["load_min", "load_max"]].isna().any(axis=1).cumsum()
        first = True
        for _, g in band.dropna().groupby(seg_id.loc[band.dropna().index]):
            if len(g) < 2:
                continue
            fig.add_trace(go.Scatter(x=pd.concat([g["time"], g["time"][::-1]]),
                                     y=pd.concat([g["load_max"], g["load_min"][::-1]]),
                                     fill="toself", fillcolor="rgba(187,134,252,0.2)", line=dict(width=0),
                                     hoverinfo="skip", name="min-max in 1 s", legendgroup="band", showlegend=first))
            first = False
    fig.add_trace(go.Scatter(x=_x(d), y=smooth(d["load_mean"], smooth_s), name="mean", line=dict(color=PURPLE, width=2)))
    fig.update_yaxes(title_text=label("load_mean"))
    if gaps is not None:
        shade_gaps(fig, gaps)
    return _style(fig, "Load cell / strain gauge", 420)


def fig_quality(df: pd.DataFrame, gaps: pd.DataFrame | None = None) -> go.Figure:
    cols = [c for c in ("missed", "imu_err", "jitter_max_us", "job_max_us", "gps_age_ms", "sats", "pi_undervolt")
            if _has(df, c)]
    if not cols:
        return empty("No quality fields")
    fig = make_subplots(rows=len(cols), cols=1, shared_xaxes=True, vertical_spacing=0.04)
    for i, c in enumerate(cols, start=1):
        bar = c in ("missed", "imu_err", "pi_undervolt")
        tr = go.Bar(x=df["time"], y=df[c], name=label(c), marker_color=RED) if bar else \
            go.Scatter(x=df["time"], y=df[c], name=label(c), line=dict(color=CYAN, width=1))
        fig.add_trace(tr, i, 1)
        fig.update_yaxes(title_text=label(c), title_font=dict(size=10), row=i, col=1)
    if gaps is not None:
        shade_gaps(fig, gaps, rows=len(cols))
    fig = _style(fig, "Data quality (red bands = missing packets)", 140 + 120 * len(cols))
    fig.update_layout(showlegend=False)
    return fig


def fig_histogram(df: pd.DataFrame, col: str, bins: int = 40) -> go.Figure:
    if not _has(df, col):
        return empty(f"No data for {col}")
    fig = go.Figure(go.Histogram(x=df[col].dropna(), nbinsx=bins, marker_color=CYAN))
    fig.update_xaxes(title_text=label(col))
    fig.update_yaxes(title_text="seconds")
    fig = _style(fig, f"Distribution of {label(col)}", 360)
    fig.update_layout(hovermode="closest", bargap=0.05)
    return fig


def fig_correlation(df: pd.DataFrame, cols: list[str] | None = None) -> go.Figure:
    cols = cols or ["speed_kmh", "long_accel_ms2", "lat_accel_ms2", "vib_rms_total", "az_rms", "az_freq",
                    "load_mean", "bme_temp", "alt_m", "batt_a"]
    cols = [c for c in cols if _has(df, c) and df[c].nunique() > 2]
    if len(cols) < 2:
        return empty("Not enough variables to correlate")
    cm = df[cols].corr().round(2)
    names = [label(c) for c in cols]
    fig = go.Figure(go.Heatmap(z=cm.values, x=names, y=names, zmin=-1, zmax=1, colorscale="RdBu_r",
                               text=cm.values, texttemplate="%{text}", colorbar=dict(thickness=12)))
    fig = _style(fig, "Correlation between variables (Pearson r)", 560)
    fig.update_layout(hovermode="closest", margin=dict(l=160, r=20, t=50, b=140))
    return fig
