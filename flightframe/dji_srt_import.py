from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

from .dji_import import _fmt_float, _parse_float

ODL_CSV_HEADERS = [
    "time_s",
    "lat",
    "lng",
    "height_m",
    "altitude_m",
    "speed_ms",
    "battery_percent",
    "satellites",
    "flight_mode",
    "rc_aileron",
    "rc_elevator",
    "rc_throttle",
    "rc_rudder",
]

_BRACKET_RE = re.compile(r"\[([^\]]+)\]")
_FONT_TAG_RE = re.compile(r"</?font[^>]*>", flags=re.IGNORECASE)


@dataclass(frozen=True)
class DjiSrtImportResult:
    srt_path: Path
    odl_csv_path: Path
    sample_count: int


@dataclass(frozen=True)
class _DjiSrtSample:
    time_s: float
    lat: float | None
    lng: float | None
    height_m: float | None
    altitude_m: float | None


def convert_dji_srt_to_odl_csv(*, input_srt: Path, output_csv: Path) -> DjiSrtImportResult:
    """
    Convert DJI embedded telemetry SRT (sidecar next to drone MP4) into overlay-ready CSV.

    Cue timestamps are used as `time_s` because DJI SRT is already aligned to the video timeline.
    GPS speed is derived from consecutive latitude/longitude samples when not present in the SRT.
    """
    if not input_srt.exists():
        raise FileNotFoundError(str(input_srt))

    text = input_srt.read_text(encoding="utf-8", errors="replace")
    samples = list(_parse_dji_srt_samples(text))
    if len(samples) < 2:
        raise ValueError("DJI SRT must contain at least 2 telemetry cues")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_odl_csv(output_csv, samples)
    return DjiSrtImportResult(srt_path=input_srt, odl_csv_path=output_csv, sample_count=len(samples))


def _parse_dji_srt_samples(text: str) -> list[_DjiSrtSample]:
    cues: list[tuple[float, dict[str, str]]] = []
    for start_s, body in _iter_srt_cues(text):
        fields = _parse_bracket_fields(body)
        if not fields:
            continue
        cues.append((start_s, fields))

    if len(cues) < 2:
        return []

    out: list[_DjiSrtSample] = []
    for time_s, fields in cues:
        out.append(
            _DjiSrtSample(
                time_s=time_s,
                lat=_parse_float(fields.get("latitude")),
                lng=_parse_float(fields.get("longitude")),
                height_m=_parse_float(fields.get("rel_alt")),
                altitude_m=_parse_float(fields.get("abs_alt")),
            )
        )

    return out


def _write_odl_csv(output_csv: Path, samples: list[_DjiSrtSample]) -> None:
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ODL_CSV_HEADERS)
        w.writeheader()

        prev_lat: float | None = None
        prev_lng: float | None = None
        prev_time: float | None = None

        for sample in samples:
            speed_ms: float | None = None
            if (
                prev_lat is not None
                and prev_lng is not None
                and sample.lat is not None
                and sample.lng is not None
                and prev_time is not None
            ):
                dt = sample.time_s - prev_time
                if dt > 0:
                    dist_m = _haversine_m(prev_lat, prev_lng, sample.lat, sample.lng)
                    speed_ms = dist_m / dt

            row = {
                "time_s": _fmt_float(sample.time_s),
                "lat": _fmt_float(sample.lat),
                "lng": _fmt_float(sample.lng),
                "height_m": _fmt_float(sample.height_m),
                "altitude_m": _fmt_float(sample.altitude_m),
                "speed_ms": _fmt_float(speed_ms),
                "battery_percent": "",
                "satellites": "",
                "flight_mode": "",
                "rc_aileron": "",
                "rc_elevator": "",
                "rc_throttle": "",
                "rc_rudder": "",
            }
            w.writerow(row)

            prev_lat, prev_lng, prev_time = sample.lat, sample.lng, sample.time_s


def _iter_srt_cues(text: str):
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue

        timing = lines[1]
        if "-->" not in timing:
            continue

        start_raw = timing.split("-->", 1)[0].strip()
        body = _FONT_TAG_RE.sub("", "\n".join(lines[2:])).strip()
        if not body:
            continue

        yield _parse_srt_timestamp(start_raw), body


def _parse_srt_timestamp(ts: str) -> float:
    ts = ts.strip()
    hours, minutes, rest = ts.split(":")
    seconds, millis = rest.split(",")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000.0


def _parse_bracket_fields(body: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for match in _BRACKET_RE.finditer(body):
        content = match.group(1).strip()
        if content.startswith("rel_alt:"):
            for key, value in re.findall(r"(\w+):\s*([^\s\]]+)", content):
                out[key] = value
            continue
        if ":" not in content:
            continue
        key, _, value = content.partition(":")
        out[key.strip()] = value.strip()
    return out


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))
