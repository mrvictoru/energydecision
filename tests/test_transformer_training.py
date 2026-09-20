import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import transformer_training as training  # noqa: E402
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


def test_build_optimizer_supports_builtin_choices():
    model = torch.nn.Linear(2, 1)

    sgd = training._build_optimizer(
        model=model,
        optimizer_name="sgd",
        lr=0.1,
        weight_decay=0.01,
        optimizer_kwargs={"momentum": 0.9},
    )
    adamw = training._build_optimizer(
        model=model,
        optimizer_name="adamw",
        lr=0.01,
        weight_decay=0.001,
        optimizer_kwargs={"eps": 1e-7},
    )

    assert isinstance(sgd, torch.optim.SGD)
    assert sgd.param_groups[0]["momentum"] == 0.9
    assert isinstance(adamw, torch.optim.AdamW)
    assert adamw.param_groups[0]["eps"] == 1e-7


def test_build_scheduler_supports_none_and_builtin_choices():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    no_scheduler = training._build_scheduler(
        optimizer=optimizer,
        scheduler_name="none",
        epochs=4,
        scheduler_kwargs=None,
    )
    cosine = training._build_scheduler(
        optimizer=optimizer,
        scheduler_name="cosineannealinglr",
        epochs=4,
        scheduler_kwargs={},
    )
    steplr = training._build_scheduler(
        optimizer=optimizer,
        scheduler_name="steplr",
        epochs=4,
        scheduler_kwargs={"step_size": 2, "gamma": 0.8},
    )

    assert isinstance(no_scheduler, training.NullLRScheduler)
    assert no_scheduler.state_dict() == {}
    assert isinstance(cosine, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert isinstance(steplr, torch.optim.lr_scheduler.StepLR)
