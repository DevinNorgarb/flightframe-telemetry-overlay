"""DJI-style corner dashboard overlay (speed ring, stats, map, G-force, heading)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import math

import cv2
import numpy as np

from .config import DashboardConfig, OverlayConfig
from .csv_parser import TelemetryData

_G = 9.80665


@dataclass
class _Trail:
    lats: np.ndarray
    lngs: np.ndarray


def draw_dji_dashboard_rgba(
    frame: np.ndarray,
    t: float,
    telemetry: TelemetryData,
    config: OverlayConfig,
) -> None:
    # Lazy import avoids circular dependency with renderer gauge helpers.
    from .renderer import _draw_gauge_rgba, _hex_to_bgra, _put_text_hud, _sample_numeric

    dc = config.dashboard
    fh, fw = frame.shape[:2]
    scale = min(fw / 1920.0, fh / 1080.0)
    margin = max(12, int(dc.margin * scale))

    label_color = _hex_to_bgra(dc.label_color_hex, 255)
    value_color = _hex_to_bgra(dc.value_color_hex, 255)
    muted_color = _hex_to_bgra(dc.muted_color_hex, 220)
    path_color = _hex_to_bgra(dc.path_color_hex, 255)
    accent_color = _hex_to_bgra(dc.accent_color_hex, 255)
    tick_color = _hex_to_bgra(config.gauges.tick_color_hex, 255)

    _draw_datetime_stack(frame, margin, margin, t, dc, scale, value_color, muted_color, _put_text_hud)
    _draw_stat_stack(frame, margin, margin, t, telemetry, config, scale, label_color, value_color, _put_text_hud, _sample_numeric)
    _draw_minimap(
        frame,
        margin,
        fh - margin - int(dc.minimap_size * scale),
        int(dc.minimap_size * scale),
        t,
        telemetry,
        scale,
        path_color,
        accent_color,
        muted_color,
        value_color,
        _put_text_hud,
        _sample_numeric,
    )
    gsize = int(dc.gforce_size * scale)
    _draw_gforce_meter(
        frame,
        fw - margin - gsize,
        margin,
        gsize,
        t,
        telemetry,
        accent_color,
        muted_color,
        value_color,
        _put_text_hud,
        _sample_numeric,
    )

    speed_size = int(dc.speed_gauge_size * scale)
    speed_val = _sample_numeric(telemetry, "speed", t)
    speed_unit = telemetry.units.get("speed", "")
    speed_max = _field_max(telemetry, "speed", 30.0)
    _draw_gauge_rgba(
        frame,
        fw - margin - speed_size,
        fh - margin - speed_size,
        speed_size,
        speed_size,
        value=speed_val,
        min_val=0.0,
        max_val=speed_max,
        label="Speed",
        unit=speed_unit,
        field="speed",
        style="hud",
        show_panel=False,
        arc_color=_hex_to_bgra(config.gauges.arc_color_hex, 255),
        needle_color=_hex_to_bgra(config.gauges.needle_color_hex, 255),
        tick_color=tick_color,
        label_color=label_color,
        value_color=value_color,
    )

    heading = _sample_numeric(telemetry, "heading_deg", t)
    if heading is None:
        heading = _estimate_heading(telemetry, t, _sample_numeric)
    if heading is not None:
        _draw_heading_badge(frame, fw // 2, fh - margin - int(36 * scale), heading, scale, value_color, _put_text_hud)


def _field_max(telemetry: TelemetryData, field: str, fallback: float) -> float:
    data = telemetry.numeric.get(field)
    if data is None:
        return fallback
    return max(float(np.max(data)) * 1.15, fallback * 0.5)


def _draw_datetime_stack(
    frame: np.ndarray,
    x: int,
    y: int,
    t: float,
    dc: DashboardConfig,
    scale: float,
    value_color: tuple[int, int, int, int],
    muted_color: tuple[int, int, int, int],
    put_text,
) -> None:
    y_cursor = y + int(22 * scale)
    if dc.start_datetime.strip():
        try:
            start = datetime.fromisoformat(dc.start_datetime.strip().replace(" ", "T", 1))
            current = start + timedelta(seconds=max(0.0, t))
            date_str = current.strftime("%Y/%m/%d")
            time_str = current.strftime("%H:%M:%S")
        except ValueError:
            date_str = "--/--/--"
            time_str = _format_elapsed(t)
    else:
        date_str = ""
        time_str = _format_elapsed(t)

    if date_str:
        put_text(frame, date_str, (x, y_cursor), font_scale=0.62 * scale, color=muted_color, thickness=1)
        y_cursor += int(28 * scale)
    put_text(frame, time_str, (x, y_cursor), font_scale=0.95 * scale, color=value_color, thickness=2)


def _format_elapsed(t: float) -> str:
    total = max(0, int(t))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _draw_stat_stack(
    frame: np.ndarray,
    x: int,
    y: int,
    t: float,
    telemetry: TelemetryData,
    config: OverlayConfig,
    scale: float,
    label_color: tuple[int, int, int, int],
    value_color: tuple[int, int, int, int],
    put_text,
    sample_numeric,
) -> None:
    dc = config.dashboard
    y_cursor = y + int(78 * scale)

    height = sample_numeric(telemetry, "height", t)
    if height is not None:
        unit = telemetry.units.get("height", "m")
        val = f"{height:.0f}" if height >= 10 else f"{height:.1f}"
        y_cursor = _draw_stat_block(frame, x, y_cursor, "Altitude", f"{val} {unit}", scale, label_color, value_color, put_text)

    slope = sample_numeric(telemetry, "pitch_deg", t)
    if slope is None:
        slope = _estimate_slope(telemetry, t, sample_numeric)
    if slope is not None:
        y_cursor = _draw_stat_block(frame, x, y_cursor, "Slope", f"{slope:.0f} deg", scale, label_color, value_color, put_text)

    distance = sample_numeric(telemetry, "distance_to_home", t)
    if distance is None:
        distance = _path_distance_m(telemetry, t)
    if distance is not None:
        unit = telemetry.units.get("distance_to_home", "m")
        if unit == "m" and distance >= 1000:
            y_cursor = _draw_stat_block(
                frame, x, y_cursor, "Distance", f"{distance / 1000:.2f} km", scale, label_color, value_color, put_text,
            )
        else:
            y_cursor = _draw_stat_block(
                frame, x, y_cursor, "Distance", f"{distance:.2f} {unit.upper()}", scale, label_color, value_color, put_text,
            )

    battery = sample_numeric(telemetry, "battery", t)
    if battery is not None and dc.show_battery:
        y_cursor = _draw_stat_block(
            frame, x, y_cursor, "Battery", f"{int(round(battery))} %", scale, label_color, value_color, put_text,
        )


def _draw_stat_block(
    frame: np.ndarray,
    x: int,
    y: int,
    label: str,
    value: str,
    scale: float,
    label_color: tuple[int, int, int, int],
    value_color: tuple[int, int, int, int],
    put_text,
) -> int:
    put_text(frame, label, (x, y), font_scale=0.40 * scale, color=label_color, thickness=1)
    put_text(frame, value, (x, y + int(30 * scale)), font_scale=0.88 * scale, color=value_color, thickness=2)
    return y + int(62 * scale)


def _estimate_slope(telemetry: TelemetryData, t: float, sample_numeric) -> float | None:
    height = sample_numeric(telemetry, "height", t)
    speed = sample_numeric(telemetry, "speed", t)
    if height is None or speed is None:
        return 0.0
    dt = 0.25
    h_prev = sample_numeric(telemetry, "height", max(0.0, t - dt))
    if h_prev is None:
        return 0.0
    climb = (height - h_prev) / dt
    horiz = max(0.5, abs(speed))
    return float(math.degrees(math.atan2(climb, horiz)))


def _path_distance_m(telemetry: TelemetryData, t: float) -> float | None:
    trail = _gps_trail(telemetry, t)
    if trail is None or len(trail.lats) < 2:
        return None
    total = 0.0
    for i in range(1, len(trail.lats)):
        total += _haversine_m(trail.lats[i - 1], trail.lngs[i - 1], trail.lats[i], trail.lngs[i])
    return total


def _gps_trail(telemetry: TelemetryData, t: float) -> _Trail | None:
    lats = telemetry.numeric.get("lat")
    lngs = telemetry.numeric.get("lng")
    if lats is None or lngs is None:
        return None
    idx = int(np.searchsorted(telemetry.time_s, t, side="right"))
    idx = max(1, min(idx, len(lats)))
    return _Trail(lats=lats[:idx], lngs=lngs[:idx])


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(d_lambda / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _draw_minimap(
    frame: np.ndarray,
    x: int,
    y: int,
    size: int,
    t: float,
    telemetry: TelemetryData,
    scale: float,
    path_color: tuple[int, int, int, int],
    accent_color: tuple[int, int, int, int],
    muted_color: tuple[int, int, int, int],
    value_color: tuple[int, int, int, int],
    put_text,
    sample_numeric,
) -> None:
    trail = _gps_trail(telemetry, t)
    pad = max(8, int(10 * scale))
    inner = size - pad * 2
    cx = x + size // 2
    cy = y + size // 2

    bg = (30, 30, 30, 90)
    cv2.rectangle(frame, (x, y), (x + size, y + size), bg, -1)
    cv2.rectangle(frame, (x, y), (x + size, y + size), muted_color, 1, cv2.LINE_AA)

    _draw_dashed_line(frame, (cx - inner // 3, cy), (cx + inner // 3, cy), muted_color)
    _draw_dashed_line(frame, (cx, cy - inner // 3), (cx, cy + inner // 3), muted_color)

    put_text(frame, "N", (cx - int(6 * scale), y + pad - int(4 * scale)), font_scale=0.38 * scale, color=value_color, thickness=1)

    if trail is None or len(trail.lats) < 2:
        put_text(frame, "NO GPS", (x + pad, cy), font_scale=0.42 * scale, color=muted_color, thickness=1)
        return

    min_lat, max_lat = float(np.min(trail.lats)), float(np.max(trail.lats))
    min_lng, max_lng = float(np.min(trail.lngs)), float(np.max(trail.lngs))
    lat_span = max(max_lat - min_lat, 1e-7)
    lng_span = max(max_lng - min_lng, 1e-7)
    span = max(lat_span, lng_span) * 1.15

    mid_lat = (min_lat + max_lat) * 0.5
    mid_lng = (min_lng + max_lng) * 0.5

    pts: list[tuple[int, int]] = []
    for lat, lng in zip(trail.lats, trail.lngs):
        px = cx + int(((lng - mid_lng) / span) * (inner * 0.45))
        py = cy - int(((lat - mid_lat) / span) * (inner * 0.45))
        pts.append((px, py))

    for i in range(1, len(pts)):
        cv2.line(frame, pts[i - 1], pts[i], path_color, max(2, int(3 * scale)), cv2.LINE_AA)

    heading = sample_numeric(telemetry, "heading_deg", t)
    if heading is None:
        heading = _estimate_heading(telemetry, t, sample_numeric) or 0.0

    tip = pts[-1]
    _draw_heading_arrow(frame, tip[0], tip[1], float(heading), max(10, int(14 * scale)), accent_color, value_color)


def _draw_dashed_line(
    frame: np.ndarray,
    p0: tuple[int, int],
    p1: tuple[int, int],
    color: tuple[int, int, int, int],
    dash: int = 6,
    gap: int = 5,
) -> None:
    x0, y0 = p0
    x1, y1 = p1
    length = max(1, int(math.hypot(x1 - x0, y1 - y0)))
    dx = (x1 - x0) / length
    dy = (y1 - y0) / length
    pos = 0
    draw = True
    while pos < length:
        seg = min(dash if draw else gap, length - pos)
        if draw:
            sx = int(x0 + dx * pos)
            sy = int(y0 + dy * pos)
            ex = int(x0 + dx * (pos + seg))
            ey = int(y0 + dy * (pos + seg))
            cv2.line(frame, (sx, sy), (ex, ey), color, 1, cv2.LINE_AA)
        pos += seg
        draw = not draw


def _draw_heading_arrow(
    frame: np.ndarray,
    x: int,
    y: int,
    heading_deg: float,
    size: int,
    fill_color: tuple[int, int, int, int],
    outline_color: tuple[int, int, int, int],
) -> None:
    ang = math.radians(heading_deg - 90.0)
    tip = (int(x + size * math.cos(ang)), int(y + size * math.sin(ang)))
    left = math.radians(heading_deg - 90.0 + 140.0)
    right = math.radians(heading_deg - 90.0 - 140.0)
    p2 = (int(x + size * 0.55 * math.cos(left)), int(y + size * 0.55 * math.sin(left)))
    p3 = (int(x + size * 0.55 * math.cos(right)), int(y + size * 0.55 * math.sin(right)))
    tri = np.array([tip, p2, p3], dtype=np.int32)
    cv2.fillConvexPoly(frame, tri, fill_color, cv2.LINE_AA)
    cv2.polylines(frame, [tri], True, outline_color, 1, cv2.LINE_AA)


def _draw_gforce_meter(
    frame: np.ndarray,
    x: int,
    y: int,
    size: int,
    t: float,
    telemetry: TelemetryData,
    accent_color: tuple[int, int, int, int],
    muted_color: tuple[int, int, int, int],
    value_color: tuple[int, int, int, int],
    put_text,
    sample_numeric,
) -> None:
    gx, gy = _sample_gforce(telemetry, t, sample_numeric)
    cx = x + size // 2
    cy = y + size // 2
    r = max(12, size // 2 - 8)

    cv2.circle(frame, (cx, cy), r, muted_color, 1, cv2.LINE_AA)
    _draw_dashed_line(frame, (cx - r, cy), (cx + r, cy), muted_color, dash=4, gap=4)
    _draw_dashed_line(frame, (cx, cy - r), (cx, cy + r), muted_color, dash=4, gap=4)

    px = int(cx + gx * r * 0.82)
    py = int(cy - gy * r * 0.82)
    cv2.circle(frame, (px, py), max(5, size // 14), accent_color, -1, cv2.LINE_AA)

    g_mag = math.sqrt(gx * gx + gy * gy)
    put_text(
        frame,
        f"{g_mag:.1f} G",
        (cx - int(22 * (size / 110)), y + size + int(18 * (size / 110))),
        font_scale=0.42 * (size / 110),
        color=value_color,
        thickness=1,
    )


def _sample_gforce(telemetry: TelemetryData, t: float, sample_numeric) -> tuple[float, float]:
    gx = sample_numeric(telemetry, "g_force_x", t)
    gy = sample_numeric(telemetry, "g_force_y", t)
    if gx is not None and gy is not None:
        return float(max(-1.5, min(1.5, gx))), float(max(-1.5, min(1.5, gy)))

    g_total = sample_numeric(telemetry, "g_force", t)
    if g_total is not None:
        return 0.0, float(max(-1.5, min(1.5, g_total - 1.0)))

    dt = 0.2
    t0 = max(float(telemetry.time_s[0]), t - dt)
    speed_now = sample_numeric(telemetry, "speed", t) or 0.0
    speed_prev = sample_numeric(telemetry, "speed", t0) or speed_now
    long_g = (speed_now - speed_prev) / max(dt, 1e-3) / _G

    head_now = sample_numeric(telemetry, "heading_deg", t)
    head_prev = sample_numeric(telemetry, "heading_deg", t0)
    lat_g = 0.0
    if head_now is not None and head_prev is not None:
        delta = ((head_now - head_prev + 180) % 360) - 180
        lat_g = math.radians(delta) * speed_now / max(dt, 1e-3) / _G

    return float(max(-1.5, min(1.5, lat_g))), float(max(-1.5, min(1.5, long_g)))


def _estimate_heading(telemetry: TelemetryData, t: float, sample_numeric) -> float | None:
    trail = _gps_trail(telemetry, t)
    if trail is None or len(trail.lats) < 2:
        return None
    lat1, lng1 = float(trail.lats[-2]), float(trail.lngs[-2])
    lat2, lng2 = float(trail.lats[-1]), float(trail.lngs[-1])
    y = math.sin(math.radians(lng2 - lng1)) * math.cos(math.radians(lat2))
    x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(
        math.radians(lat2)
    ) * math.cos(math.radians(lng2 - lng1))
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _draw_heading_badge(
    frame: np.ndarray,
    cx: int,
    y: int,
    heading: float,
    scale: float,
    color: tuple[int, int, int, int],
    put_text,
) -> None:
    text = f"{int(round(heading))} deg"
    font_scale = 0.72 * scale
    thickness = 2
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)[0]
    pad_x = int(16 * scale)
    pad_y = int(10 * scale)
    w = size[0] + pad_x * 2
    h = size[1] + pad_y * 2
    x0 = cx - w // 2
    slant = int(8 * scale)

    pts = np.array(
        [
            [x0 + slant, y],
            [x0 + w, y],
            [x0 + w - slant, y + h],
            [x0, y + h],
        ],
        dtype=np.int32,
    )
    fill = (20, 20, 20, 140)
    cv2.fillConvexPoly(frame, pts, fill, cv2.LINE_AA)
    cv2.polylines(frame, [pts], True, (255, 255, 255, 180), 1, cv2.LINE_AA)
    put_text(frame, text, (x0 + pad_x, y + pad_y + size[1] - 2), font_scale=font_scale, color=color, thickness=thickness)
