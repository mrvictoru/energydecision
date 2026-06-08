from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress_bar import ProgressBar
    from rich.table import Table
    from rich.text import Text

    RICH_IMPORT_ERROR: Exception | None = None
    RICH_AVAILABLE = True
except ImportError as exc:  # pragma: no cover - exercised via fallback tests
    Console = None  # type: ignore[assignment]
    Group = None  # type: ignore[assignment]
    Layout = None  # type: ignore[assignment]
    Live = None  # type: ignore[assignment]
    Panel = None  # type: ignore[assignment]
    ProgressBar = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]
    Text = None  # type: ignore[assignment]
    RICH_IMPORT_ERROR = exc
    RICH_AVAILABLE = False

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    PSUTIL_AVAILABLE = False

SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"
DEFAULT_HISTORY_SECONDS = 600
DEFAULT_POLL_SECONDS = 2.0

TQDM_LINE_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)/(?P<epochs>\d+):\s+"
    r"(?P<pct>[\d.]+)%\|.*?\|\s*"
    r"(?P<batch>\d+)/(?P<total>\d+)\s*"
    r"\[(?P<elapsed>[\d:]+)<(?P<remaining>[\d:]+),\s*"
    r"(?P<rate>[\d.]+)batch/s,\s*"
    r"(?P<rest>.+)\]"
)


def parse_tqdm_line(line: str) -> dict[str, Any] | None:
    match = TQDM_LINE_RE.search(line)
    if not match:
        return None
    rest = match.group("rest")
    result: dict[str, Any] = {
        "epoch": int(match.group("epoch")),
        "epochs": int(match.group("epochs")),
        "progress_pct": float(match.group("pct")),
        "batch": int(match.group("batch")),
        "total_batches": int(match.group("total")),
        "elapsed": match.group("elapsed"),
        "remaining": match.group("remaining"),
        "batch_per_s": float(match.group("rate")),
    }
    for key in ("loss", "avg", "ema", "lr"):
        m = re.search(rf"\b{key}=([\d.eE+\-]+)", rest)
        if m:
            try:
                result[key] = float(m.group(1))
            except ValueError:
                pass
    for key in ("skip", "seg"):
        m = re.search(rf"\b{key}=(\d+)", rest)
        if m:
            result[key] = int(m.group(1))
    for key in ("cpu", "gpu", "pcpu", "pth"):
        m = re.search(rf"\b{key}=(\d+)%?", rest)
        if m:
            result[key] = int(m.group(1))
    for key in ("ram", "vram", "vpeak", "prss", "pvms"):
        m = re.search(rf"\b{key}=([\d.]+)/([\d.]+)G", rest)
        if m:
            result[f"{key}_used"] = float(m.group(1))
            result[f"{key}_total"] = float(m.group(2))
        else:
            m = re.search(rf"\b{key}=([\d.]+)G", rest)
            if m:
                result[f"{key}_used"] = float(m.group(1))
    return result


@dataclass(frozen=True)
class DashboardState:
    command_text: str
    elapsed_text: str
    log_path_text: str
    surface_text: str | None
    dataset_text: str | None
    run_summary_text: str | None
    status: str
    child_state: str | None
    exit_code: int | None
    progress_text: str
    progress_percent: float | None
    train_text: str
    val_text: str
    best_text: str
    resources_text: str
    last_checkpoint_text: str | None
    log_tail: list[str]
    system_metrics: dict[str, Any] = field(default_factory=dict)
    loss_history: list[float] = field(default_factory=list)
    val_loss_history: list[float] = field(default_factory=list)
    gpu_util_history: list[float] = field(default_factory=list)
    gpu_temp_history: list[float] = field(default_factory=list)
    gpu_power_history: list[float] = field(default_factory=list)
    csv_path: str | None = None
    config_text: str | None = None
    config_rows: list[tuple[str, str]] = field(default_factory=list)


class SystemMonitor:
    def __init__(self, refresh_seconds: float = 2.0):
        self.refresh_seconds = max(0.5, float(refresh_seconds))
        self._last_snapshot: dict[str, Any] = {}
        self._last_sample_time: float = 0.0
        self._nvidia_smi_available: bool = True
        self._psutil_available: bool = PSUTIL_AVAILABLE
        self._gpu_name: str | None = None
        self._prev_disk: tuple[int, int] | None = None
        self._prev_disk_time: float = 0.0

    def poll(self, *, force: bool = False) -> dict[str, Any]:
        now = time.monotonic()
        if not force and self._last_snapshot and (now - self._last_sample_time) < self.refresh_seconds:
            return dict(self._last_snapshot)
        metrics: dict[str, Any] = {}
        metrics.update(self._poll_psutil())
        metrics.update(self._poll_nvidia_smi())
        self._last_snapshot = metrics
        self._last_sample_time = now
        return dict(metrics)

    def _poll_psutil(self) -> dict[str, Any]:
        if not self._psutil_available:
            return {}
        metrics: dict[str, Any] = {}
        try:
            metrics["sys_cpu_pct"] = psutil.cpu_percent(interval=0)
            vm = psutil.virtual_memory()
            metrics["sys_ram_used_gb"] = round(vm.used / (1024**3), 1)
            metrics["sys_ram_total_gb"] = round(vm.total / (1024**3), 1)
        except Exception:
            self._psutil_available = False
            return metrics
        try:
            dio = psutil.disk_io_counters()
            if dio is not None:
                now = time.monotonic()
                current = (dio.read_bytes, dio.write_bytes)
                if self._prevDisk is not None and self._prev_disk_time > 0:
                    dt = now - self._prev_disk_time
                    if dt > 0:
                        read_rate = (current[0] - self._prev_disk[0]) / dt / (1024**2)
                        write_rate = (current[1] - self._prev_disk[1]) / dt / (1024**2)
                        metrics["disk_read_mbs"] = round(max(0.0, read_rate), 1)
                        metrics["disk_write_mbs"] = round(max(0.0, write_rate), 1)
                self._prev_disk = current
                self._prev_disk_time = now
        except Exception:
            pass
        return metrics

    def _poll_nvidia_smi(self) -> dict[str, Any]:
        if not self._nvidia_smi_available:
            return {}
        command = [
            "nvidia-smi",
            "--query-gpu=name,temperature.gpu,power.draw,fan.speed,clocks.sm,clocks.mem,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
        except (FileNotFoundError, OSError, subprocess.SubprocessError):
            self._nvidia_smi_available = False
            return {}
        lines = completed.stdout.strip().splitlines()
        if not lines:
            self._nvidia_smi_available = False
            return {}
        parts = [p.strip() for p in lines[0].split(",")]
        if len(parts) < 9:
            self._nvidia_smi_available = False
            return {}

        def _safe_float(val: str) -> float | None:
            try:
                v = float(val.strip())
                return v if v == v else None
            except (ValueError, TypeError):
                return None

        def _safe_int(val: str) -> int | None:
            try:
                return int(val.strip())
            except (ValueError, TypeError):
                return None

        metrics: dict[str, Any] = {}
        name = parts[0].strip()
        if name and "[Unknown]" not in name:
            metrics["gpu_name"] = name
            self._gpu_name = name
        elif self._gpu_name:
            metrics["gpu_name"] = self._gpu_name
        for key, parser in [
            ("gpu_temp_c", _safe_int),
            ("gpu_power_w", _safe_float),
            ("gpu_fan_pct", _safe_int),
            ("gpu_core_mhz", _safe_int),
            ("gpu_mem_mhz", _safe_int),
            ("gpu_util_pct", _safe_int),
        ]:
            val = parser(parts[[
                "gpu_temp_c", "gpu_power_w", "gpu_fan_pct",
                "gpu_core_mhz", "gpu_mem_mhz", "gpu_util_pct",
            ].index(key) + 1])
            if val is not None:
                metrics[key] = round(val, 1) if isinstance(val, float) else val
        vram_used = _safe_float(parts[7])
        vram_total = _safe_float(parts[8])
        if vram_used is not None:
            metrics["vram_used_gb"] = round(vram_used / 1024, 1)
        if vram_total is not None:
            metrics["vram_total_gb"] = round(vram_total / 1024, 1)
        return metrics


class MetricsHistory:
    CSV_COLUMNS = [
        "timestamp", "epoch", "segment", "progress_fraction",
        "train_loss", "val_loss", "log_loss", "best_score",
        "sys_cpu_pct", "sys_ram_used_gb", "sys_ram_total_gb",
        "disk_read_mbs", "disk_write_mbs",
        "gpu_util_pct", "gpu_temp_c", "gpu_power_w", "gpu_fan_pct",
        "gpu_core_mhz", "gpu_mem_mhz",
        "vram_used_gb", "vram_total_gb",
    ]

    def __init__(self, max_samples: int = 300, csv_path: Path | None = None):
        self.max_samples = max_samples
        self.csv_path = csv_path
        self._buffer: deque[dict[str, Any]] = deque(maxlen=max_samples)
        self._csv_fh: Any = None
        self._csv_writer: Any = None
        if csv_path is not None:
            try:
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                self._csv_fh = open(csv_path, "w", newline="", encoding="utf-8")
                self._csv_writer = csv.DictWriter(
                    self._csv_fh, fieldnames=self.CSV_COLUMNS, extrasaction="ignore",
                )
                self._csv_writer.writeheader()
            except OSError:
                self._csv_fh = None
                self._csv_writer = None

    def record(self, *, snapshot: dict[str, Any] | None, system: dict[str, Any], parsed: dict[str, Any] | None = None) -> None:
        row: dict[str, Any] = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        if snapshot:
            row["epoch"] = snapshot.get("epoch", "")
            row["segment"] = snapshot.get("segment", "")
            row["progress_fraction"] = snapshot.get("progress_fraction", "")
            ct = snapshot.get("current_train") or {}
            row["train_loss"] = ct.get("train_total_avg", "")
            val = snapshot.get("validation") or {}
            row["val_loss"] = val.get("val_total", "")
            best = snapshot.get("best") or {}
            row["best_score"] = best.get("score", "")
        if parsed is not None:
            row["epoch"] = row.get("epoch") or parsed.get("epoch", "")
            row["segment"] = row.get("segment") or parsed.get("seg", "")
            if not row.get("progress_fraction"):
                row["progress_fraction"] = parsed.get("progress_pct", "") if isinstance(parsed.get("progress_pct"), (int, float)) else ""
            if not row.get("train_loss") and parsed.get("loss") is not None:
                row["train_loss"] = parsed["loss"]
            row["log_loss"] = parsed.get("loss", "")
        row.update(system)
        self._buffer.append(row)
        if self._csv_writer is not None:
            try:
                self._csv_writer.writerow(row)
                if self._csv_fh:
                    self._csv_fh.flush()
            except OSError:
                pass

    def history(self, key: str) -> list[float]:
        values: list[float] = []
        for sample in self._buffer:
            v = sample.get(key)
            if isinstance(v, (int, float)):
                values.append(float(v))
        return values

    def close(self) -> None:
        if self._csv_fh is not None:
            try:
                self._csv_fh.close()
            except OSError:
                pass


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or attach to a DT training dashboard from its snapshot file.",
    )
    parser.add_argument(
        "--progress-snapshot-path",
        type=Path,
        required=True,
        help="Path to the live JSON progress snapshot written by the training process.",
    )
    parser.add_argument(
        "--surface-manifest-path",
        type=Path,
        default=None,
        help="Optional training-surface manifest JSON to enrich the dashboard.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Where to tee the child process output. Defaults next to the progress snapshot.",
    )
    parser.add_argument(
        "--attach",
        action="store_true",
        help="Attach to an already-running training process instead of launching a child command.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=DEFAULT_POLL_SECONDS,
        help="How often to refresh the dashboard.",
    )
    parser.add_argument(
        "--tail-lines",
        type=int,
        default=8,
        help="How many lines of log tail to display.",
    )
    parser.add_argument(
        "--ui",
        choices=["auto", "plain", "rich"],
        default="auto",
        help="Terminal UI mode. 'auto' prefers Rich on TTYs and falls back to plain text elsewhere.",
    )
    parser.add_argument(
        "--history-seconds",
        type=float,
        default=DEFAULT_HISTORY_SECONDS,
        help="Seconds of history to keep for sparklines (default: 600 = 10 min).",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Disable CSV metrics logging.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to execute after -- (omit in --attach mode).")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def tail_lines(path: Path, limit: int) -> list[str]:
    if limit <= 0:
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            return list(deque(fh, maxlen=limit))
    except OSError:
        return []


def format_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    seconds = max(0, int(round(value)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def format_row(mapping: dict[str, Any], keys: list[str]) -> str:
    parts: list[str] = []
    for key in keys:
        value = mapping.get(key)
        if value in (None, ""):
            continue
        parts.append(f"{key}={value}")
    return " | ".join(parts) if parts else "n/a"


def sparkline(values: list[float], width: int | None = None) -> str:
    if not values:
        return ""
    if width is not None and len(values) > width:
        values = values[-width:]
    lo, hi = min(values), max(values)
    span = hi - lo
    n = len(SPARKLINE_CHARS) - 1
    if span == 0:
        return SPARKLINE_CHARS[n // 2] * len(values)
    return "".join(
        SPARKLINE_CHARS[min(n, max(0, int((v - lo) / span * n)))]
        for v in values
    )


def bar(fraction: float, width: int = 20) -> str:
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    return "█" * filled + "░" * (width - filled)


def normalize_command(command: list[str], *, attach: bool) -> list[str]:
    normalized = list(command)
    if normalized and normalized[0] == "--":
        normalized = normalized[1:]
    if attach:
        if normalized:
            raise ValueError("No command may be provided when using --attach.")
        return []
    if not normalized:
        raise ValueError("A command must be provided after --.")
    return normalized


def progress_percent(snapshot: dict[str, Any]) -> float | None:
    progress_fraction = snapshot.get("progress_fraction")
    if isinstance(progress_fraction, (int, float)):
        return max(0.0, min(100.0, float(progress_fraction) * 100.0))
    epochs = snapshot.get("epochs")
    if not isinstance(epochs, int) or epochs <= 0:
        return None
    epoch = snapshot.get("epoch")
    if not isinstance(epoch, int):
        return None
    checkpoints_per_epoch = snapshot.get("checkpoints_per_epoch")
    segment = snapshot.get("segment")
    if isinstance(segment, int) and segment >= 0 and isinstance(checkpoints_per_epoch, int) and checkpoints_per_epoch > 0:
        completed = (epoch - 1) + ((segment + 1) / checkpoints_per_epoch)
    else:
        completed = epoch - 1
    if snapshot.get("status") == "finished":
        completed = float(epochs)
    return max(0.0, min(100.0, (completed / epochs) * 100.0))


def _parse_last_tqdm_line(log_tail: list[str]) -> dict[str, Any] | None:
    for line in reversed(log_tail):
        parsed = parse_tqdm_line(line)
        if parsed is not None:
            return parsed
    return None


def build_dashboard_state(
    *,
    snapshot: dict[str, Any] | None,
    manifest: dict[str, Any] | None,
    log_path: Path,
    log_tail: list[str],
    command: list[str],
    child: subprocess.Popen[Any] | None,
    started_at: float,
    return_code: int | None = None,
    system_metrics: dict[str, Any] | None = None,
    loss_history: list[float] | None = None,
    val_loss_history: list[float] | None = None,
    gpu_util_history: list[float] | None = None,
    gpu_temp_history: list[float] | None = None,
    gpu_power_history: list[float] | None = None,
    csv_path: str | None = None,
) -> DashboardState:
    surface_text: str | None = None
    dataset_text: str | None = None
    run_summary_text: str | None = None
    if manifest:
        surface = {
            "preset": manifest.get("surface_preset"),
            "variant": manifest.get("model_variant"),
            "optimizer": manifest.get("optimizer"),
            "scheduler": manifest.get("scheduler"),
        }
        surface_text = format_row(surface, ["preset", "variant", "optimizer", "scheduler"])
        if surface_text == "n/a":
            surface_text = None
        dataset_summary = manifest.get("dataset_summary") or {}
        train_dataset = dataset_summary.get("train") or {}
        val_dataset = dataset_summary.get("val") or {}
        dataset_text = format_row(
            {
                "train_files": train_dataset.get("file_count"),
                "train_eps": train_dataset.get("episode_count"),
                "train_windows": train_dataset.get("window_count"),
                "val_files": val_dataset.get("file_count"),
                "val_eps": val_dataset.get("episode_count"),
                "val_windows": val_dataset.get("window_count"),
            },
            ["train_files", "train_eps", "train_windows", "val_files", "val_eps", "val_windows"],
        )
        if dataset_text == "n/a":
            dataset_text = None
        run_summary = manifest.get("run_summary") or {}
        run_summary_text = format_row(
            {
                "wins_per_s": run_summary.get("effective_windows_per_second"),
                "elapsed_s": run_summary.get("elapsed_seconds"),
                "checkpoints": run_summary.get("checkpoint_count"),
            },
            ["wins_per_s", "elapsed_s", "checkpoints"],
        )
        if run_summary_text == "n/a":
            run_summary_text = None

    config_text = None
    config_rows: list[tuple[str, str]] = []
    if manifest:
        mk = manifest.get("model_kwargs") or {}
        config_pairs: list[tuple[str, str]] = []
        for key, label, source in [
            ("batch_size", "batch", None),
            ("context_len", "context", mk),
            ("n_block", "n_block", mk),
            ("h_dim", "h_dim", mk),
            ("n_heads", "n_heads", mk),
            ("drop_p", "drop_p", mk),
            ("act_dim", "act_dim", mk),
            ("state_dim", "state_dim", mk),
        ]:
            val = source.get(key) if source else None
            if val is not None:
                config_rows.append((label, str(val)))
        variant = manifest.get("model_variant")
        if variant:
            config_rows.append(("variant", str(variant)))
        optimizer = manifest.get("optimizer")
        if optimizer:
            config_rows.append(("opt", str(optimizer)))
        scheduler = manifest.get("scheduler")
        if scheduler:
            config_rows.append(("sched", str(scheduler)))
        ckpt_epoch = manifest.get("checkpoints_per_epoch")
        if ckpt_epoch:
            config_rows.append(("ckpt/epoch", str(ckpt_epoch)))
        train_w = (manifest.get("dataset_summary") or {}).get("train", {}).get("window_count")
        if train_w:
            config_rows.append(("windows", f"{train_w:,}"))
        config_text = "  ".join(f"{k}={v}" for k, v in config_rows[:12])

    child_state: str | None = None
    if child is not None:
        child_state = "running" if child.poll() is None else "stopped"

    parsed = _parse_last_tqdm_line(log_tail)
    log_loss = parsed.get("loss") if parsed else None
    if log_loss is not None and loss_history is not None:
        loss_history = list(loss_history) + [log_loss]

    if snapshot is None:
        if parsed is not None:
            status = "running"
            progress = parsed.get("progress_pct")
            epoch = parsed.get("epoch", "?")
            epochs = parsed.get("epochs", "?")
            segment = parsed.get("seg", "?")
            batch = parsed.get("batch", 0)
            total = parsed.get("total_batches", 0)
            loss_val = parsed.get("loss")
            avg_val = parsed.get("avg")
            ema_val = parsed.get("ema")
            lr_val = parsed.get("lr")
            elapsed = parsed.get("elapsed", "?")
            remaining = parsed.get("remaining", "?")
            rate = parsed.get("batch_per_s", 0)
            progress_text = f"epoch {epoch}/{epochs} seg {segment}  |  batch {batch}/{total}  |  {progress:.1f}%  |  {elapsed}<{remaining}  |  {rate:.2f} batch/s"
            train_parts: list[str] = []
            if loss_val is not None:
                train_parts.append(f"loss={loss_val:.4f}")
            if avg_val is not None:
                train_parts.append(f"avg={avg_val:.4f}")
            if ema_val is not None:
                train_parts.append(f"ema={ema_val:.4f}")
            if lr_val is not None:
                train_parts.append(f"lr={lr_val:.2e}")
            train_text = "  ".join(train_parts) if train_parts else "n/a"
        else:
            status = "waiting for training output"
            progress = None
            progress_text = "waiting for training output"
            train_text = "n/a"
        return DashboardState(
            command_text=" ".join(command) if command else "(attach mode)",
            elapsed_text=format_seconds(time.monotonic() - started_at),
            log_path_text=str(log_path),
            surface_text=surface_text,
            dataset_text=dataset_text,
            run_summary_text=run_summary_text,
            status=status,
            child_state=child_state,
            exit_code=return_code,
            progress_text=progress_text,
            progress_percent=progress,
            train_text=train_text,
            val_text="n/a",
            best_text="n/a",
            resources_text="n/a",
            last_checkpoint_text=None,
            log_tail=[line.rstrip() for line in log_tail] or ["(no log output yet)"],
            system_metrics=system_metrics or {},
            loss_history=loss_history or [],
            val_loss_history=val_loss_history or [],
            gpu_util_history=gpu_util_history or [],
            gpu_temp_history=gpu_temp_history or [],
            gpu_power_history=gpu_power_history or [],
            csv_path=csv_path,
            config_text=config_text,
            config_rows=config_rows,
        )

    progress = progress_percent(snapshot)
    current_train = snapshot.get("current_train") or {}
    validation = snapshot.get("validation") or {}
    best = snapshot.get("best") or {}
    resources = snapshot.get("resources") or {}
    latest_history = snapshot.get("latest_history") or []
    epoch = snapshot.get("epoch", "?")
    epochs = snapshot.get("epochs", "?")
    segment = snapshot.get("segment", "?")
    last_checkpoint_text: str | None = None
    if latest_history:
        last_checkpoint_text = format_row(
            latest_history[-1],
            ["epoch", "segment", "train_total_avg", "val_total", "train_total_ema"],
        )

    val_loss = validation.get("val_total")
    if val_loss is not None and val_loss_history is not None:
        val_loss_history = list(val_loss_history) + [float(val_loss)]

    if parsed is not None:
        p_epoch = parsed.get("epoch", "?")
        p_epochs = parsed.get("epochs", "?")
        p_seg = parsed.get("seg", "?")
        p_batch = parsed.get("batch", 0)
        p_total = parsed.get("total_batches", 0)
        p_progress = parsed.get("progress_pct", 0)
        p_elapsed = parsed.get("elapsed", "?")
        p_remaining = parsed.get("remaining", "?")
        p_rate = parsed.get("batch_per_s", 0)
        progress_text = f"epoch {p_epoch}/{p_epochs} seg {p_seg}  |  batch {p_batch}/{p_total}  |  {p_progress:.1f}%  |  {p_elapsed}<{p_remaining}  |  {p_rate:.2f} batch/s"
        train_parts = []
        if parsed.get("loss") is not None:
            train_parts.append(f"loss={parsed['loss']:.4f}")
        if parsed.get("avg") is not None:
            train_parts.append(f"avg={parsed['avg']:.4f}")
        if parsed.get("ema") is not None:
            train_parts.append(f"ema={parsed['ema']:.4f}")
        if parsed.get("lr") is not None:
            train_parts.append(f"lr={parsed['lr']:.2e}")
        train_text = "  ".join(train_parts) if train_parts else format_row(
            current_train,
            ["train_total_avg", "train_action_avg", "train_state_avg", "train_return_avg", "train_total_ema"],
        )
    else:
        progress_text = (
            f"progress: {progress:.1f}%"
            if progress is not None
            else f"progress: epoch {epoch}/{epochs} seg {segment}"
        )
        train_text = format_row(
            current_train,
            ["train_total_avg", "train_action_avg", "train_state_avg", "train_return_avg", "train_total_ema"],
        )

    return DashboardState(
        command_text=" ".join(command) if command else "(attach mode)",
        elapsed_text=format_seconds(time.monotonic() - started_at),
        log_path_text=str(log_path),
        surface_text=surface_text,
        dataset_text=dataset_text,
        run_summary_text=run_summary_text,
        status=str(snapshot.get("status", "unknown")),
        child_state=child_state,
        exit_code=return_code,
        progress_text=progress_text,
        progress_percent=progress,
        train_text=train_text,
        val_text=format_row(validation, ["val_total", "val_action", "val_state", "val_return", "val_valid"]),
        best_text=format_row(best, ["score", "val_loss", "train_loss_est"]),
        resources_text=format_row(resources, ["cpu", "ram", "gpu", "vram", "vpeak", "pcpu", "prss", "pvms", "pth"]),
        last_checkpoint_text=last_checkpoint_text,
        log_tail=[line.rstrip() for line in log_tail] or ["(no log output yet)"],
        system_metrics=system_metrics or {},
        loss_history=loss_history or [],
        val_loss_history=val_loss_history or [],
        gpu_util_history=gpu_util_history or [],
        gpu_temp_history=gpu_temp_history or [],
        gpu_power_history=gpu_power_history or [],
        csv_path=csv_path,
        config_text=config_text,
        config_rows=config_rows,
    )


def render_plain_dashboard(state: DashboardState) -> str:
    sm = state.system_metrics
    lines: list[str] = []
    lines.append("DT progress runner")
    lines.append(f"command: {state.command_text}")
    lines.append(f"elapsed: {state.elapsed_text}")
    lines.append(f"log: {state.log_path_text}")
    if state.surface_text:
        lines.append(f"surface: {state.surface_text}")
    if state.dataset_text:
        lines.append(f"datasets: {state.dataset_text}")
    status_line = f"status: {state.status}"
    if state.child_state:
        status_line += f" | child={state.child_state}"
    if state.exit_code is not None:
        status_line += f" | exit={state.exit_code}"
    lines.append(status_line)
    lines.append(state.progress_text)
    lines.append(f"train: {state.train_text}")
    lines.append(f"val: {state.val_text}")
    lines.append(f"best: {state.best_text}")
    lines.append(f"resources: {state.resources_text}")
    if sm:
        gpu_parts: list[str] = []
        if sm.get("gpu_name"):
            gpu_parts.append(str(sm["gpu_name"]))
        if sm.get("gpu_temp_c") is not None:
            gpu_parts.append(f"{sm['gpu_temp_c']}°C")
        if sm.get("gpu_power_w") is not None:
            gpu_parts.append(f"{sm['gpu_power_w']}W")
        if sm.get("gpu_fan_pct") is not None:
            gpu_parts.append(f"fan {sm['gpu_fan_pct']}%")
        if sm.get("gpu_util_pct") is not None:
            gpu_parts.append(f"util {sm['gpu_util_pct']}%")
        if sm.get("vram_used_gb") is not None and sm.get("vram_total_gb") is not None:
            gpu_parts.append(f"vram {sm['vram_used_gb']}/{sm['vram_total_gb']}G")
        if gpu_parts:
            lines.append(f"gpu: {' | '.join(gpu_parts)}")
        sys_parts: list[str] = []
        if sm.get("sys_cpu_pct") is not None:
            sys_parts.append(f"cpu {sm['sys_cpu_pct']}%")
        if sm.get("sys_ram_used_gb") is not None and sm.get("sys_ram_total_gb") is not None:
            sys_parts.append(f"ram {sm['sys_ram_used_gb']}/{sm['sys_ram_total_gb']}G")
        if sm.get("disk_read_mbs") is not None:
            sys_parts.append(f"disk R:{sm['disk_read_mbs']} W:{sm.get('disk_write_mbs', '?')} MB/s")
        if sys_parts:
            lines.append(f"system: {' | '.join(sys_parts)}")
    if state.loss_history:
        lines.append(f"train loss: {sparkline(state.loss_history, width=60)}  ({min(state.loss_history):.4f} → {max(state.loss_history):.4f})")
    if state.val_loss_history:
        lines.append(f"  val loss: {sparkline(state.val_loss_history, width=60)}  ({min(state.val_loss_history):.4f} → {max(state.val_loss_history):.4f})")
    if state.gpu_util_history:
        lines.append(f"gpu util:   {sparkline(state.gpu_util_history, width=60)}")
    if state.gpu_temp_history:
        lines.append(f"gpu temp:   {sparkline(state.gpu_temp_history, width=60)}  ({min(state.gpu_temp_history):.0f}–{max(state.gpu_temp_history):.0f}°C)")
    if state.run_summary_text:
        lines.append(f"run summary: {state.run_summary_text}")
    if state.last_checkpoint_text is not None:
        lines.append(f"last checkpoint: {state.last_checkpoint_text}")
    if state.config_text:
        lines.append(f"config: {state.config_text}")
    if state.csv_path:
        lines.append(f"csv: {state.csv_path}")
    lines.append("log tail:")
    lines.extend(f"  {line}" for line in state.log_tail)
    return "\n".join(lines)


def render_dashboard(
    *,
    snapshot: dict[str, Any] | None,
    manifest: dict[str, Any] | None,
    log_path: Path,
    log_tail: list[str],
    command: list[str],
    child: subprocess.Popen[Any] | None,
    started_at: float,
    return_code: int | None = None,
    system_metrics: dict[str, Any] | None = None,
    loss_history: list[float] | None = None,
    val_loss_history: list[float] | None = None,
    gpu_util_history: list[float] | None = None,
    gpu_temp_history: list[float] | None = None,
    gpu_power_history: list[float] | None = None,
    csv_path: str | None = None,
) -> str:
    state = build_dashboard_state(
        snapshot=snapshot,
        manifest=manifest,
        log_path=log_path,
        log_tail=log_tail,
        command=command,
        child=child,
        started_at=started_at,
        return_code=return_code,
        system_metrics=system_metrics,
        loss_history=loss_history,
        val_loss_history=val_loss_history,
        gpu_util_history=gpu_util_history,
        gpu_temp_history=gpu_temp_history,
        gpu_power_history=gpu_power_history,
        csv_path=csv_path,
    )
    return render_plain_dashboard(state)


def resolve_ui_mode(ui: str, *, is_tty: bool) -> str:
    if ui == "plain":
        return "plain"
    if ui == "rich":
        return "rich" if RICH_AVAILABLE and is_tty else "plain"
    return "rich" if RICH_AVAILABLE and is_tty else "plain"


def rich_status_style(status: str) -> str:
    normalized = status.lower()
    if normalized == "finished":
        return "bold green"
    if normalized in {"running", "starting"}:
        return "bold cyan"
    if normalized in {"failed", "error", "crash"}:
        return "bold red"
    return "bold yellow"


def _temp_style(temp_c: float | None) -> str:
    if temp_c is None:
        return "dim"
    if temp_c >= 85:
        return "bold red"
    if temp_c >= 70:
        return "yellow"
    return "green"


def _power_style(power_w: float | None, max_power: float = 200.0) -> str:
    if power_w is None:
        return "dim"
    ratio = power_w / max_power
    if ratio >= 0.9:
        return "bold red"
    if ratio >= 0.7:
        return "yellow"
    return "green"


def build_rich_dashboard(state: DashboardState) -> Any:
    if not RICH_AVAILABLE:
        raise RuntimeError(f"Rich rendering requested but unavailable: {RICH_IMPORT_ERROR}")

    sm = state.system_metrics

    header = Table.grid(expand=True)
    header.add_column(ratio=3)
    header.add_column(ratio=2, justify="right")
    status_icon = "●" if state.status in {"running", "starting"} else ("✗" if state.status in {"failed", "error", "crash"} else ("✓" if state.status == "finished" else "○"))
    status_text = Text.assemble(
        ("DT progress runner", "bold"),
        "  ",
        (status_icon, rich_status_style(state.status)),
        " ",
        (state.status, rich_status_style(state.status)),
    )
    right_bits = [f"elapsed {state.elapsed_text}"]
    if state.child_state:
        right_bits.append(f"child {state.child_state}")
    if state.exit_code is not None:
        right_bits.append(f"exit {state.exit_code}")
    header.add_row(status_text, Text(" | ".join(right_bits), style="dim"))

    progress_items: list[Any] = []
    epoch_info = state.progress_text
    progress_items.append(Text(epoch_info, style="bold"))
    if state.progress_percent is not None:
        progress_items.append(
            Text(f"  {bar(state.progress_percent / 100, width=30)}  {state.progress_percent:.1f}%", style="bold cyan")
        )

    if state.train_text and state.train_text != "n/a":
        progress_items.append(Text(state.train_text, style=""))

    if state.loss_history and len(state.loss_history) >= 2:
        spark_text = sparkline(state.loss_history, width=50)
        lo, hi = min(state.loss_history), max(state.loss_history)
        progress_items.append(Text(""))
        progress_items.append(Text(f" train loss [{len(state.loss_history)} samples]", style="green"))
        progress_items.append(Text(f" {spark_text}", style="green"))
        progress_items.append(Text(f" {lo:.4f} → {hi:.4f}  (Δ {hi - lo:.4f})", style="dim"))

    if state.val_loss_history and len(state.val_loss_history) >= 2:
        v_spark = sparkline(state.val_loss_history, width=50)
        v_lo, v_hi = min(state.val_loss_history), max(state.val_loss_history)
        progress_items.append(Text(""))
        progress_items.append(Text(f" val loss   [{len(state.val_loss_history)} samples]", style="yellow"))
        progress_items.append(Text(f" {v_spark}", style="yellow"))
        progress_items.append(Text(f" {v_lo:.4f} → {v_hi:.4f}  (Δ {v_hi - v_lo:.4f})", style="dim"))

    if state.val_text and state.val_text != "n/a":
        progress_items.append(Text(""))
        progress_items.append(Text(state.val_text, style=""))
    if state.best_text and state.best_text != "n/a":
        progress_items.append(Text(state.best_text, style="bold green"))
    if state.last_checkpoint_text:
        progress_items.append(Text(state.last_checkpoint_text, style="dim"))

    gpu_lines: list[Any] = []
    gpu_name = sm.get("gpu_name", "GPU")
    gpu_lines.append(Text(str(gpu_name), style="bold magenta"))

    temp = sm.get("gpu_temp_c")
    power = sm.get("gpu_power_w")
    fan = sm.get("gpu_fan_pct")
    core = sm.get("gpu_core_mhz")
    mem_clk = sm.get("gpu_mem_mhz")

    info_parts: list[str] = []
    if temp is not None:
        info_parts.append(f"{temp}°C")
    if power is not None:
        info_parts.append(f"{power:.0f}W")
    if fan is not None:
        info_parts.append(f"fan {fan}%")
    if info_parts:
        line = Text(" ")
        line.append("  ".join(info_parts))
        gpu_lines.append(line)

    clk_parts: list[str] = []
    if core is not None:
        clk_parts.append(f"core {core} MHz")
    if mem_clk is not None:
        clk_parts.append(f"mem {mem_clk} MHz")
    if clk_parts:
        gpu_lines.append(Text(f" {'  '.join(clk_parts)}", style="dim"))

    gpu_util = sm.get("gpu_util_pct")
    if gpu_util is not None:
        gpu_lines.append(Text(f" util {bar(gpu_util / 100, width=12)} {gpu_util}%", style="cyan"))

    vram_used = sm.get("vram_used_gb")
    vram_total = sm.get("vram_total_gb")
    if vram_used is not None and vram_total is not None and vram_total > 0:
        vram_frac = vram_used / vram_total
        gpu_lines.append(Text(f" vram {bar(vram_frac, width=12)} {vram_used}/{vram_total}G", style="magenta"))

    if state.gpu_util_history and len(state.gpu_util_history) >= 2:
        gpu_lines.append(Text(""))
        gpu_lines.append(Text(f" {sparkline(state.gpu_util_history, width=24)}", style="cyan"))
    if state.gpu_temp_history and len(state.gpu_temp_history) >= 2:
        lo_t, hi_t = min(state.gpu_temp_history), max(state.gpu_temp_history)
        gpu_lines.append(Text(f" {sparkline(state.gpu_temp_history, width=24)}  {lo_t:.0f}–{hi_t:.0f}°C", style=_temp_style(hi_t)))

    sys_lines: list[Any] = []
    sys_lines.append(Text("System", style="bold"))
    sys_cpu = sm.get("sys_cpu_pct")
    if sys_cpu is not None:
        sys_lines.append(Text(f" CPU  {bar(sys_cpu / 100, width=12)} {sys_cpu:.0f}%", style="cyan"))
    sys_ram = sm.get("sys_ram_used_gb")
    sys_ram_total = sm.get("sys_ram_total_gb")
    if sys_ram is not None and sys_ram_total is not None and sys_ram_total > 0:
        ram_frac = sys_ram / sys_ram_total
        sys_lines.append(Text(f" RAM  {bar(ram_frac, width=12)} {sys_ram:.1f}/{sys_ram_total:.1f}G", style="magenta"))
    disk_r = sm.get("disk_read_mbs")
    disk_w = sm.get("disk_write_mbs")
    if disk_r is not None:
        sys_lines.append(Text(f" Disk R:{disk_r:.1f} W:{disk_w or 0:.1f} MB/s", style="dim"))
    res_text = state.resources_text
    if res_text and res_text != "n/a":
        sys_lines.append(Text(""))
        sys_lines.append(Text("Process", style="bold"))
        sys_lines.append(Text(f" {res_text}", style="dim"))

    config_panel_items: list[Any] = []
    if state.config_rows:
        config_table = Table.grid(padding=(0, 2))
        config_table.add_column(style="bold cyan", no_wrap=True)
        config_table.add_column(style="")
        for key, val in state.config_rows:
            config_table.add_row(key, str(val))
        config_panel_items.append(config_table)
    elif state.surface_text or state.dataset_text or state.run_summary_text:
        if state.surface_text:
            config_panel_items.append(Text(f" surface: {state.surface_text}", style="dim"))
        if state.dataset_text:
            config_panel_items.append(Text(f" datasets: {state.dataset_text}", style="dim"))
        if state.run_summary_text:
            config_panel_items.append(Text(f" run: {state.run_summary_text}", style="dim"))
    if state.csv_path:
        config_panel_items.append(Text(f" csv: {state.csv_path}", style="dim"))

    log_text = Text("\n".join(state.log_tail))

    layout = Layout()
    layout.split_column(
        Layout(Panel(header, title="status", border_style="bright_blue"), size=3),
        Layout(name="body", ratio=4),
        Layout(Panel(log_text, title="log tail", border_style="yellow"), ratio=1),
        Layout(Panel(Text("q: ctrl-c / stop process | --ui plain for fallback", style="dim"), border_style="grey50"), size=3),
    )
    layout["body"].split_row(
        Layout(name="left", ratio=3),
        Layout(name="right", ratio=3),
    )
    layout["left"].split_column(
        Layout(Panel(Group(*progress_items), title="training progress", border_style="green"), ratio=1),
    )
    layout["right"].split_column(
        Layout(Panel(Group(*gpu_lines), title="gpu", border_style="magenta"), ratio=2),
        Layout(Panel(Group(*sys_lines), title="system", border_style="cyan"), ratio=2),
        Layout(Panel(Group(*config_panel_items), title="config", border_style="blue"), ratio=2),
    )
    return layout


def maybe_warn_plain_fallback(*, requested_ui: str, resolved_ui: str) -> None:
    if resolved_ui != "plain":
        return
    if requested_ui == "rich" and not RICH_AVAILABLE:
        print(f"[dt_progress_runner] Falling back to plain UI because rich is unavailable: {RICH_IMPORT_ERROR}", file=sys.stderr)
    elif requested_ui == "rich" and not sys.stdout.isatty():
        print("[dt_progress_runner] Falling back to plain UI because stdout is not a TTY.", file=sys.stderr)
    elif requested_ui == "auto" and not RICH_AVAILABLE:
        print("[dt_progress_runner] rich not installed; using plain UI.", file=sys.stderr)


def _compute_max_samples(history_seconds: float, poll_seconds: float) -> int:
    return max(10, int(history_seconds / max(0.1, poll_seconds)))


def run_plain_loop(
    *,
    snapshot_path: Path,
    manifest_path: Path,
    log_path: Path,
    command: list[str],
    child: subprocess.Popen[Any] | None,
    started_at: float,
    poll_seconds: float,
    tail_limit: int,
    system_monitor: SystemMonitor | None = None,
    metrics_history: MetricsHistory | None = None,
    csv_path_str: str | None = None,
) -> int:
    last_render: str | None = None
    return_code: int | None = None
    try:
        while True:
            snapshot = load_json(snapshot_path)
            manifest = load_json(manifest_path)
            log_tail = tail_lines(log_path, tail_limit)
            sys_metrics: dict[str, Any] = {}
            parsed: dict[str, Any] | None = _parse_last_tqdm_line(log_tail)
            if system_monitor is not None:
                sys_metrics = system_monitor.poll()
            if metrics_history is not None:
                metrics_history.record(snapshot=snapshot, system=sys_metrics, parsed=parsed)
            lh = metrics_history.history("train_loss") if metrics_history else []
            vh = metrics_history.history("val_loss") if metrics_history else []
            render = render_dashboard(
                snapshot=snapshot,
                manifest=manifest,
                log_path=log_path,
                log_tail=log_tail,
                command=command,
                child=child,
                started_at=started_at,
                system_metrics=sys_metrics,
                loss_history=lh,
                val_loss_history=vh,
                gpu_util_history=metrics_history.history("gpu_util_pct") if metrics_history else [],
                gpu_temp_history=metrics_history.history("gpu_temp_c") if metrics_history else [],
                gpu_power_history=metrics_history.history("gpu_power_w") if metrics_history else [],
                csv_path=csv_path_str,
            )
            if render != last_render:
                if sys.stdout.isatty():
                    sys.stdout.write("\033[2J\033[H")
                print(render, flush=True)
                last_render = render
            if child is not None and child.poll() is not None:
                break
            time.sleep(max(0.2, poll_seconds))
    finally:
        if child is not None:
            return_code = child.wait()
        if metrics_history is not None:
            metrics_history.close()

    final_snapshot = load_json(snapshot_path)
    final_manifest = load_json(manifest_path)
    final_sys: dict[str, Any] = {}
    if system_monitor is not None:
        final_sys = system_monitor.poll(force=True)
    final_render = render_dashboard(
        snapshot=final_snapshot,
        manifest=final_manifest,
        log_path=log_path,
        log_tail=tail_lines(log_path, tail_limit),
        command=command,
        child=child,
        started_at=started_at,
        return_code=return_code,
        system_metrics=final_sys,
        loss_history=metrics_history.history("train_loss") if metrics_history else [],
        val_loss_history=metrics_history.history("val_loss") if metrics_history else [],
        gpu_util_history=metrics_history.history("gpu_util_pct") if metrics_history else [],
        gpu_temp_history=metrics_history.history("gpu_temp_c") if metrics_history else [],
        gpu_power_history=metrics_history.history("gpu_power_w") if metrics_history else [],
        csv_path=csv_path_str,
    )
    if sys.stdout.isatty():
        sys.stdout.write("\033[2J\033[H")
    print(final_render, flush=True)
    return return_code or 0


def run_rich_loop(
    *,
    snapshot_path: Path,
    manifest_path: Path,
    log_path: Path,
    command: list[str],
    child: subprocess.Popen[Any] | None,
    started_at: float,
    poll_seconds: float,
    tail_limit: int,
    system_monitor: SystemMonitor | None = None,
    metrics_history: MetricsHistory | None = None,
    csv_path_str: str | None = None,
) -> int:
    console = Console()
    return_code: int | None = None
    with Live(console=console, screen=True, auto_refresh=False) as live:
        try:
            while True:
                snapshot = load_json(snapshot_path)
                manifest = load_json(manifest_path)
                log_tail = tail_lines(log_path, tail_limit)
                sys_metrics: dict[str, Any] = {}
                parsed: dict[str, Any] | None = _parse_last_tqdm_line(log_tail)
                if system_monitor is not None:
                    sys_metrics = system_monitor.poll()
                if metrics_history is not None:
                    metrics_history.record(snapshot=snapshot, system=sys_metrics, parsed=parsed)
                lh = metrics_history.history("train_loss") if metrics_history else []
                vh = metrics_history.history("val_loss") if metrics_history else []
                state = build_dashboard_state(
                    snapshot=snapshot,
                    manifest=manifest,
                    log_path=log_path,
                    log_tail=log_tail,
                    command=command,
                    child=child,
                    started_at=started_at,
                    system_metrics=sys_metrics,
                    loss_history=lh,
                    val_loss_history=vh,
                    gpu_util_history=metrics_history.history("gpu_util_pct") if metrics_history else [],
                    gpu_temp_history=metrics_history.history("gpu_temp_c") if metrics_history else [],
                    gpu_power_history=metrics_history.history("gpu_power_w") if metrics_history else [],
                    csv_path=csv_path_str,
                )
                live.update(build_rich_dashboard(state), refresh=True)
                if child is not None and child.poll() is not None:
                    break
                time.sleep(max(0.2, poll_seconds))
        finally:
            if child is not None:
                return_code = child.wait()
            if metrics_history is not None:
                metrics_history.close()

        final_snapshot = load_json(snapshot_path)
        final_manifest = load_json(manifest_path)
        final_sys: dict[str, Any] = {}
        if system_monitor is not None:
            final_sys = system_monitor.poll(force=True)
        final_state = build_dashboard_state(
            snapshot=final_snapshot,
            manifest=final_manifest,
            log_path=log_path,
            log_tail=tail_lines(log_path, tail_limit),
            command=command,
            child=child,
            started_at=started_at,
            return_code=return_code,
            system_metrics=final_sys,
            loss_history=metrics_history.history("train_loss") if metrics_history else [],
            val_loss_history=metrics_history.history("val_loss") if metrics_history else [],
            gpu_util_history=metrics_history.history("gpu_util_pct") if metrics_history else [],
            gpu_temp_history=metrics_history.history("gpu_temp_c") if metrics_history else [],
            gpu_power_history=metrics_history.history("gpu_power_w") if metrics_history else [],
            csv_path=csv_path_str,
        )
        live.update(build_rich_dashboard(final_state), refresh=True)
    return return_code or 0


def main() -> int:
    args = parse_args()
    command = normalize_command(list(args.command), attach=bool(args.attach))

    snapshot_path = args.progress_snapshot_path.resolve()
    manifest_path = (
        args.surface_manifest_path.resolve()
        if args.surface_manifest_path is not None
        else snapshot_path.with_name(snapshot_path.stem.replace("_progress", "_surface_manifest") + ".json")
    )
    log_path = (
        args.log_path.resolve()
        if args.log_path is not None
        else snapshot_path.with_name(snapshot_path.stem.replace("_progress", "_monitor") + ".log")
    )
    if not args.attach:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_ui = resolve_ui_mode(args.ui, is_tty=sys.stdout.isatty())
    maybe_warn_plain_fallback(requested_ui=args.ui, resolved_ui=resolved_ui)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    csv_path: Path | None = None
    csv_path_str: str | None = None
    if not args.no_csv:
        csv_path = snapshot_path.with_name("dashboard_metrics.csv")
        csv_path_str = str(csv_path)

    max_samples = _compute_max_samples(args.history_seconds, args.poll_seconds)
    metrics_history: MetricsHistory | None = None
    if csv_path is not None:
        metrics_history = MetricsHistory(max_samples=max_samples, csv_path=csv_path)
    else:
        metrics_history = MetricsHistory(max_samples=max_samples, csv_path=None)

    system_monitor = SystemMonitor(refresh_seconds=args.poll_seconds)

    started_at = time.monotonic()
    child: subprocess.Popen[Any] | None = None
    if not args.attach:
        with log_path.open("w", encoding="utf-8") as log_file:
            child = subprocess.Popen(
                command,
                cwd=repo_root(),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
            )

            if resolved_ui == "rich":
                return run_rich_loop(
                    snapshot_path=snapshot_path,
                    manifest_path=manifest_path,
                    log_path=log_path,
                    command=command,
                    child=child,
                    started_at=started_at,
                    poll_seconds=float(args.poll_seconds),
                    tail_limit=args.tail_lines,
                    system_monitor=system_monitor,
                    metrics_history=metrics_history,
                    csv_path_str=csv_path_str,
                )
            return run_plain_loop(
                snapshot_path=snapshot_path,
                manifest_path=manifest_path,
                log_path=log_path,
                command=command,
                child=child,
                started_at=started_at,
                poll_seconds=float(args.poll_seconds),
                tail_limit=args.tail_lines,
                system_monitor=system_monitor,
                metrics_history=metrics_history,
                csv_path_str=csv_path_str,
            )

    if resolved_ui == "rich":
        return run_rich_loop(
            snapshot_path=snapshot_path,
            manifest_path=manifest_path,
            log_path=log_path,
            command=command,
            child=child,
            started_at=started_at,
            poll_seconds=float(args.poll_seconds),
            tail_limit=args.tail_lines,
            system_monitor=system_monitor,
            metrics_history=metrics_history,
            csv_path_str=csv_path_str,
        )
    return run_plain_loop(
        snapshot_path=snapshot_path,
        manifest_path=manifest_path,
        log_path=log_path,
        command=command,
        child=child,
        started_at=started_at,
        poll_seconds=float(args.poll_seconds),
        tail_limit=args.tail_lines,
        system_monitor=system_monitor,
        metrics_history=metrics_history,
        csv_path_str=csv_path_str,
    )


if __name__ == "__main__":
    raise SystemExit(main())
