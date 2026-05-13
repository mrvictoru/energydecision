from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a DT training command and render a live progress dashboard from its snapshot file.",
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
        "--poll-seconds",
        type=float,
        default=2.0,
        help="How often to refresh the dashboard.",
    )
    parser.add_argument(
        "--tail-lines",
        type=int,
        default=8,
        help="How many lines of log tail to display.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to execute after --.")
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
) -> str:
    lines: list[str] = []
    lines.append("DT progress runner")
    lines.append(f"command: {' '.join(command)}")
    lines.append(f"elapsed: {format_seconds(time.monotonic() - started_at)}")
    lines.append(f"log: {log_path}")

    if manifest:
        surface = {
            "preset": manifest.get("surface_preset"),
            "variant": manifest.get("model_variant"),
            "optimizer": manifest.get("optimizer"),
            "scheduler": manifest.get("scheduler"),
        }
        lines.append(f"surface: {format_row(surface, ['preset', 'variant', 'optimizer', 'scheduler'])}")

    if snapshot is None:
        lines.append("status: waiting for progress snapshot")
    else:
        progress = progress_percent(snapshot)
        status = snapshot.get("status", "unknown")
        epoch = snapshot.get("epoch", "?")
        epochs = snapshot.get("epochs", "?")
        segment = snapshot.get("segment", "?")
        current_train = snapshot.get("current_train") or {}
        validation = snapshot.get("validation") or {}
        best = snapshot.get("best") or {}
        resources = snapshot.get("resources") or {}
        latest_history = snapshot.get("latest_history") or []
        lines.append(
            f"status: {status}"
            + (f" | child={'running' if child and child.poll() is None else 'stopped'}" if child else "")
            + (f" | exit={return_code}" if return_code is not None else "")
        )
        lines.append(
            f"progress: {progress:.1f}%" if progress is not None else f"progress: epoch {epoch}/{epochs} seg {segment}"
        )
        lines.append(
            "train: "
            + format_row(
                current_train,
                ["train_total_avg", "train_action_avg", "train_state_avg", "train_return_avg", "train_total_ema"],
            )
        )
        lines.append(
            "val: "
            + format_row(validation, ["val_total", "val_action", "val_state", "val_return", "val_valid"])
        )
        lines.append("best: " + format_row(best, ["score", "val_loss", "train_loss_est"]))
        lines.append("resources: " + format_row(resources, ["cpu", "ram", "gpu", "vram", "vpeak", "pcpu", "prss", "pvms", "pth"]))
        if latest_history:
            last = latest_history[-1]
            lines.append(
                "last checkpoint: "
                + format_row(last, ["epoch", "segment", "train_total_avg", "val_total", "train_total_ema"])
            )

    lines.append("log tail:")
    if log_tail:
        lines.extend(f"  {line.rstrip()}" for line in log_tail)
    else:
        lines.append("  (no log output yet)")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("A command must be provided after --.")

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
    log_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with log_path.open("w", encoding="utf-8") as log_file:
        child = subprocess.Popen(
            command,
            cwd=repo_root(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )

        started_at = time.monotonic()
        last_render: str | None = None
        try:
            while True:
                snapshot = load_json(snapshot_path)
                manifest = load_json(manifest_path)
                log_tail = tail_lines(log_path, args.tail_lines)
                render = render_dashboard(
                    snapshot=snapshot,
                    manifest=manifest,
                    log_path=log_path,
                    log_tail=log_tail,
                    command=command,
                    child=child,
                    started_at=started_at,
                )
                if render != last_render:
                    if sys.stdout.isatty():
                        sys.stdout.write("\033[2J\033[H")
                    print(render, flush=True)
                    last_render = render
                if child.poll() is not None:
                    break
                time.sleep(max(0.2, float(args.poll_seconds)))
        finally:
            return_code = child.wait()

    final_snapshot = load_json(snapshot_path)
    final_manifest = load_json(manifest_path)
    final_render = render_dashboard(
        snapshot=final_snapshot,
        manifest=final_manifest,
        log_path=log_path,
        log_tail=tail_lines(log_path, args.tail_lines),
        command=command,
        child=child,
        started_at=started_at,
        return_code=return_code,
    )
    if sys.stdout.isatty():
        sys.stdout.write("\033[2J\033[H")
    print(final_render, flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
