from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest

from flightframe.csv_parser import load_telemetry
from flightframe.dji_srt_import import convert_dji_srt_to_odl_csv


SAMPLE_SRT = """\
1
00:00:00,000 --> 00:00:00,020
<font size="28">FrameCnt: 1, DiffTime: 20ms
2026-06-06 19:28:02.476
[iso: 5590] [shutter: 1/50.0] [fnum: 1.8] [ev: -0.7] [color_md: default] [focal_len: 24.00] [latitude: -33.866016] [longitude: 18.512237] [rel_alt: 110.500 abs_alt: 63.692] [ct: 4916] </font>

2
00:00:00,020 --> 00:00:00,040
<font size="28">FrameCnt: 2, DiffTime: 20ms
2026-06-06 19:28:02.497
[iso: 5590] [shutter: 1/50.0] [fnum: 1.8] [ev: -0.7] [color_md: default] [focal_len: 24.00] [latitude: -33.866116] [longitude: 18.512337] [rel_alt: 110.600 abs_alt: 63.792] [ct: 4918] </font>

3
00:00:00,040 --> 00:00:00,061
<font size="28">FrameCnt: 3, DiffTime: 21ms
2026-06-06 19:28:02.518
[iso: 5590] [shutter: 1/50.0] [fnum: 1.8] [ev: -0.7] [color_md: default] [focal_len: 24.00] [latitude: -33.866216] [longitude: 18.512437] [rel_alt: 110.700 abs_alt: 63.892] [ct: 4918] </font>
"""


def _write_srt(content: str) -> Path:
    p = Path(tempfile.NamedTemporaryFile(suffix=".SRT", delete=False).name)
    p.write_text(content, encoding="utf-8")
    return p


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestDjiSrtImport:
    def test_maps_cue_timestamps_and_fields(self):
        srt = _write_srt(SAMPLE_SRT)
        out = Path(tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name)
        res = convert_dji_srt_to_odl_csv(input_srt=srt, output_csv=out)
        assert res.sample_count == 3

        rows = _read_rows(out)
        assert rows[0]["time_s"] == "0"
        assert rows[1]["time_s"] == "0.02"
        assert rows[2]["time_s"] == "0.04"
        assert rows[0]["lat"] == "-33.866016"
        assert rows[0]["lng"] == "18.512237"
        assert rows[0]["height_m"] == "110.5"
        assert rows[0]["altitude_m"] == "63.692"

    def test_derives_speed_from_gps(self):
        srt = _write_srt(SAMPLE_SRT)
        out = Path(tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name)
        convert_dji_srt_to_odl_csv(input_srt=srt, output_csv=out)
        rows = _read_rows(out)
        assert rows[0]["speed_ms"] == ""
        assert float(rows[1]["speed_ms"]) > 0

    def test_loads_through_csv_parser(self):
        srt = _write_srt(SAMPLE_SRT)
        out = Path(tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name)
        convert_dji_srt_to_odl_csv(input_srt=srt, output_csv=out)
        telemetry = load_telemetry(out)
        assert len(telemetry.time_s) == 3
        assert "lat" in telemetry.numeric
        assert "height" in telemetry.numeric
        assert "speed" in telemetry.numeric

    def test_raises_on_empty_srt(self):
        srt = _write_srt("1\n00:00:00,000 --> 00:00:00,020\n[latitude: 1] [longitude: 2]\n")
        out = Path(tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name)
        with pytest.raises(ValueError, match="at least 2"):
            convert_dji_srt_to_odl_csv(input_srt=srt, output_csv=out)
