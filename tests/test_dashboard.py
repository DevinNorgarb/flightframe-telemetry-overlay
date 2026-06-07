from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np

from flightframe.config import OverlayConfig, load_config
from flightframe.csv_parser import load_telemetry
from flightframe.renderer import _draw_overlay_rgba


def _write_csv(rows: list[dict]) -> Path:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    writer = csv.DictWriter(tmp, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    tmp.close()
    return Path(tmp.name)


class TestDashboardRender:
    def test_dashboard_renders_without_crash(self):
        rows = [
            {
                "time_s": 0.0,
                "speed_ms": 5.0,
                "height_m": 40.0,
                "battery_percent": 90,
                "lat": 37.8000,
                "lng": -122.4000,
                "heading_deg": 168.0,
            },
            {
                "time_s": 1.0,
                "speed_ms": 12.0,
                "height_m": 48.0,
                "battery_percent": 88,
                "lat": 37.8005,
                "lng": -122.3995,
                "heading_deg": 170.0,
            },
            {
                "time_s": 2.0,
                "speed_ms": 18.0,
                "height_m": 55.0,
                "battery_percent": 86,
                "lat": 37.8010,
                "lng": -122.3990,
                "heading_deg": 172.0,
            },
        ]
        telemetry = load_telemetry(_write_csv(rows))

        cfg = OverlayConfig()
        cfg.dashboard.enabled = True
        cfg.rc_sticks.enabled = False
        cfg.gauges.enabled = False
        cfg.transparent_output.width = 1280
        cfg.transparent_output.height = 720

        frame = np.zeros((720, 1280, 4), dtype=np.uint8)
        frame = _draw_overlay_rgba(frame, 1.0, telemetry, cfg)
        assert frame.shape == (720, 1280, 4)
        assert frame[:, :, 3].max() > 0

    def test_dashboard_example_config_loads(self):
        cfg = load_config(Path("examples/dashboard.config.yaml"))
        assert cfg.dashboard.enabled is True
        assert cfg.gauges.enabled is False

    def test_dashboard_optional_classic_panel(self):
        rows = [
            {"time_s": 0.0, "speed_ms": 1.0, "height_m": 10.0, "lat": 1.0, "lng": 2.0},
            {"time_s": 1.0, "speed_ms": 2.0, "height_m": 11.0, "lat": 1.1, "lng": 2.1},
        ]
        telemetry = load_telemetry(_write_csv(rows))
        cfg = OverlayConfig()
        cfg.dashboard.enabled = True
        cfg.dashboard.show_classic_panel = False

        frame_hud = _draw_overlay_rgba(np.zeros((480, 640, 4), dtype=np.uint8), 0.5, telemetry, cfg)

        cfg.dashboard.show_classic_panel = True
        frame_with_panel = _draw_overlay_rgba(np.zeros((480, 640, 4), dtype=np.uint8), 0.5, telemetry, cfg)
        assert not np.array_equal(frame_hud, frame_with_panel)
