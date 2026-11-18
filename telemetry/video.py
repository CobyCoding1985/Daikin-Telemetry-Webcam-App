"""
video.py - Session video recorded by telemetria_final.py (v2.3+):
  session_X_video.mjpeg   JPEG frames back to back
  session_X_frames.csv    frame, pi_time, offset, size

Frames are found by byte offset (instant, no decoding of the whole file, no ffmpeg/OpenCV).
export_mp4() converts a range to a normal MP4 (H.264) with the ffmpeg shipped in imageio-ffmpeg.
They are put on the same time axis as the CSV through the Pi clock, which both files share.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd


def read_frames_index(source) -> pd.DataFrame:
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    fr = pd.read_csv(source)
    fr["pi_time"] = pd.to_datetime(fr["pi_time"], errors="coerce")
    return fr.dropna(subset=["pi_time"]).sort_values("pi_time").reset_index(drop=True)


def align(frames: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Add 'time' (session time base) and 'elapsed_s' to the frame index.
    The session time may be GPS UTC; both files carry the Pi clock, so the offset between them
    is measured on the CSV and applied to the frames."""
    fr = frames.copy()
    offset = pd.Timedelta(0)
    if "pi_time" in df.columns and len(df):
        pt = pd.to_datetime(df["pi_time"], errors="coerce")
        diff = (df["time"] - pt).dropna()
        if len(diff):
            offset = diff.median()
    fr["time"] = fr["pi_time"] + offset
    t0 = df["time"].min() if len(df) else fr["time"].min()
    fr["elapsed_s"] = (fr["time"] - t0).dt.total_seconds()
    return fr


class FrameReader:
    """Reads one JPEG by offset from a path (seek) or from bytes already in memory."""

    def __init__(self, source):
        self.path = source if isinstance(source, str) else None
        self.data = None if self.path else bytes(source)

    def get(self, offset: int, size: int) -> bytes:
        if self.data is not None:
            return self.data[offset: offset + size]
        with open(self.path, "rb") as fh:
            fh.seek(int(offset))
            return fh.read(int(size))


def nearest(frames: pd.DataFrame, elapsed_s: float) -> int:
    i = int(np.searchsorted(frames["elapsed_s"].to_numpy(), elapsed_s))
    i = min(max(i, 0), len(frames) - 1)
    if i > 0 and abs(frames["elapsed_s"].iloc[i - 1] - elapsed_s) < abs(frames["elapsed_s"].iloc[i] - elapsed_s):
        i -= 1
    return i


def row_at(df: pd.DataFrame, t: pd.Timestamp, max_gap_s: float = 1.5) -> pd.Series | None:
    """Telemetry row closest in time to t (None if the closest one is too far away)."""
    if df.empty:
        return None
    i = int(np.argmin(np.abs((df["time"] - t).dt.total_seconds().to_numpy())))
    if abs((df["time"].iloc[i] - t).total_seconds()) > max_gap_s:
        return None
    return df.iloc[i]


def hud(jpeg: bytes, lines: list[str], fmt: str = "PNG") -> bytes:
    """Burn a small text overlay (speed, vibration...) onto the frame. Returns PNG (or JPEG) bytes."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.open(io.BytesIO(jpeg)).convert("RGB")
    w, h = im.size
    scale = max(1, w // 320)
    try:
        font = ImageFont.load_default(size=11 * scale)
    except TypeError:                      # Pillow < 10.1
        font = ImageFont.load_default()
    d = ImageDraw.Draw(im, "RGBA")
    line_h = 14 * scale
    d.rectangle([0, h - line_h * len(lines) - 6, w, h], fill=(0, 0, 0, 150))
    for k, text in enumerate(lines):
        d.text((6, h - line_h * (len(lines) - k) - 3), text, fill=(255, 255, 255, 255), font=font)
    out = io.BytesIO()
    if fmt.upper() == "JPEG":
        im.save(out, "JPEG", quality=90)
    else:
        im.save(out, "PNG")
    return out.getvalue()


def export_mp4(reader: FrameReader, frames: pd.DataFrame, out_path: str, fps: float,
               overlay=None, progress=None) -> str:
    """Write frames (rows of the frame index) to an H.264 MP4 that plays everywhere.
    overlay: optional function(frame_row) -> list[str] burned onto each frame.
    progress: optional function(fraction 0..1). Uses the ffmpeg bundled with imageio-ffmpeg."""
    import subprocess

    import imageio_ffmpeg
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "mjpeg", "-framerate", f"{fps:.3f}", "-i", "-",
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    n = len(frames)
    try:
        for k, (_, f) in enumerate(frames.iterrows()):
            jp = reader.get(f["offset"], f["size"])
            if overlay is not None:
                jp = hud(jp, overlay(f), fmt="JPEG")
            proc.stdin.write(jp)
            if progress is not None and (k % 25 == 0 or k == n - 1):
                progress((k + 1) / n)
        proc.stdin.close()
    except BrokenPipeError:
        pass
    err = proc.stderr.read().decode(errors="replace")
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed: {err.strip()[:300]}")
    return out_path
