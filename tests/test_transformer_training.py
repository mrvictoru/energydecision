import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transformer_training import TrainingResourceMonitor  # noqa: E402


GIB = 1024**3


def test_training_resource_monitor_formats_resource_snapshot(monkeypatch):
    monitor = TrainingResourceMonitor(refresh_seconds=0.0)

    monkeypatch.setattr(monitor, "_sample_cpu_percent", lambda: 42.4)
    monkeypatch.setattr(monitor, "_sample_ram_usage", lambda: (4 * GIB, 8 * GIB))
    monkeypatch.setattr(
        monitor,
        "_sample_gpu_stats",
        lambda device: {
            "utilization_percent": 77.2,
            "vram_used_bytes": 2 * GIB,
            "vram_total_bytes": 10 * GIB,
            "peak_vram_bytes": 3 * GIB,
        },
    )
    monkeypatch.setattr(
        monitor,
        "_sample_process_stats",
        lambda: {
            "pcpu": "18%",
            "prss": "1.5G",
            "pvms": "3.0G",
            "pth": "12",
        },
    )

    snapshot = monitor.snapshot(device="cuda:0")

    assert snapshot == {
        "cpu": "42%",
        "ram": "4.0/8.0G",
        "gpu": "77%",
        "vram": "2.0/10.0G",
        "vpeak": "3.0G",
        "pcpu": "18%",
        "prss": "1.5G",
        "pvms": "3.0G",
        "pth": "12",
    }


def test_training_resource_monitor_reuses_cached_snapshot(monkeypatch):
    monitor = TrainingResourceMonitor(refresh_seconds=60.0)
    call_count = {"cpu": 0}

    def fake_cpu():
        call_count["cpu"] += 1
        return 55.0

    monkeypatch.setattr(monitor, "_sample_cpu_percent", fake_cpu)
    monkeypatch.setattr(monitor, "_sample_ram_usage", lambda: (6 * GIB, 12 * GIB))
    monkeypatch.setattr(monitor, "_sample_gpu_stats", lambda device: None)
    monkeypatch.setattr(monitor, "_sample_process_stats", lambda: {"pcpu": "21%"})

    first = monitor.snapshot(device="cpu")
    second = monitor.snapshot(device="cpu")

    assert first == second
    assert first["cpu"] == "55%"
    assert first["ram"] == "6.0/12.0G"
    assert first["pcpu"] == "21%"
    assert call_count["cpu"] == 1
