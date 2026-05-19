from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
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


@dataclass(frozen=True)
class DashboardState:
    command_text: str
    elapsed_text: str
    log_path_text: str
    surface_text: str | None
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
        default=2.0,
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
) -> DashboardState:
    surface_text: str | None = None
    if manifest:
        surface = {
            "preset": manifest.get("surface_preset"),
            "variant": manifest.get("model_variant"),
            "optimizer": manifest.get("optimizer"),
            "scheduler": manifest.get("scheduler"),
        }
        surface_text = format_row(surface, ["preset", "variant", "optimizer", "scheduler"])

    child_state: str | None = None
    if child is not None:
        child_state = "running" if child.poll() is None else "stopped"

    if snapshot is None:
        return DashboardState(
            command_text=" ".join(command) if command else "(attach mode)",
            elapsed_text=format_seconds(time.monotonic() - started_at),
            log_path_text=str(log_path),
            surface_text=surface_text,
            status="waiting for progress snapshot",
            child_state=child_state,
            exit_code=return_code,
            progress_text="progress: waiting for snapshot",
            progress_percent=None,
            train_text="n/a",
            val_text="n/a",
            best_text="n/a",
            resources_text="n/a",
            last_checkpoint_text=None,
            log_tail=[line.rstrip() for line in log_tail] or ["(no log output yet)"],
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

    return DashboardState(
        command_text=" ".join(command) if command else "(attach mode)",
        elapsed_text=format_seconds(time.monotonic() - started_at),
        log_path_text=str(log_path),
        surface_text=surface_text,
        status=str(snapshot.get("status", "unknown")),
        child_state=child_state,
        exit_code=return_code,
        progress_text=(
            f"progress: {progress:.1f}%"
            if progress is not None
            else f"progress: epoch {epoch}/{epochs} seg {segment}"
        ),
        progress_percent=progress,
        train_text=format_row(
            current_train,
            ["train_total_avg", "train_action_avg", "train_state_avg", "train_return_avg", "train_total_ema"],
        ),
        val_text=format_row(validation, ["val_total", "val_action", "val_state", "val_return", "val_valid"]),
        best_text=format_row(best, ["score", "val_loss", "train_loss_est"]),
        resources_text=format_row(resources, ["cpu", "ram", "gpu", "vram", "vpeak", "pcpu", "prss", "pvms", "pth"]),
        last_checkpoint_text=last_checkpoint_text,
        log_tail=[line.rstrip() for line in log_tail] or ["(no log output yet)"],
    )


def render_plain_dashboard(state: DashboardState) -> str:
    lines: list[str] = []
    lines.append("DT progress runner")
    lines.append(f"command: {state.command_text}")
    lines.append(f"elapsed: {state.elapsed_text}")
    lines.append(f"log: {state.log_path_text}")
    if state.surface_text:
        lines.append(f"surface: {state.surface_text}")
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
    if state.last_checkpoint_text is not None:
        lines.append(f"last checkpoint: {state.last_checkpoint_text}")
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


def build_rich_dashboard(state: DashboardState) -> Any:
    if not RICH_AVAILABLE:
        raise RuntimeError(f"Rich rendering requested but unavailable: {RICH_IMPORT_ERROR}")

    header = Table.grid(expand=True)
    header.add_column(ratio=3)
    header.add_column(ratio=2, justify="right")
    status_text = Text.assemble(("DT progress runner", "bold"), "  ", (state.status, rich_status_style(state.status)))
    right_bits = [f"elapsed {state.elapsed_text}"]
    if state.child_state:
        right_bits.append(f"child {state.child_state}")
    if state.exit_code is not None:
        right_bits.append(f"exit {state.exit_code}")
    header.add_row(status_text, Text(" | ".join(right_bits), style="dim"))

    summary = Table.grid(padding=(0, 1))
    summary.add_column(style="bold cyan", no_wrap=True)
    summary.add_column()
    summary.add_row("command", state.command_text)
    summary.add_row("log", state.log_path_text)
    if state.surface_text:
        summary.add_row("surface", state.surface_text)

    metrics = Table.grid(padding=(0, 1))
    metrics.add_column(style="bold cyan", no_wrap=True)
    metrics.add_column()
    metrics.add_row("progress", state.progress_text)
    metrics.add_row("train", state.train_text)
    metrics.add_row("val", state.val_text)
    metrics.add_row("best", state.best_text)
    if state.last_checkpoint_text:
        metrics.add_row("last", state.last_checkpoint_text)

    progress_group_items: list[Any] = [summary]
    if state.progress_percent is not None:
        progress_group_items.extend(
            [
                Text("training progress", style="bold"),
                ProgressBar(total=100, completed=state.progress_percent, width=None),
            ]
        )
    progress_group_items.append(metrics)

    resources = Table.grid(padding=(0, 1))
    resources.add_column(style="bold cyan", no_wrap=True)
    resources.add_column()
    resources.add_row("resources", state.resources_text)
    resources.add_row("ui", "rich" if RICH_AVAILABLE else "plain")

    history_lines: list[Any] = [Text(state.last_checkpoint_text or "n/a")]
    history_group = Group(*history_lines)

    log_text = Text("\n".join(state.log_tail))

    layout = Layout()
    layout.split_column(
        Layout(Panel(header, title="status", border_style="bright_blue"), size=3),
        Layout(name="body", ratio=1),
        Layout(Panel(Text("q: ctrl-c / stop process | --ui plain for fallback", style="dim"), border_style="grey50"), size=3),
    )
    layout["body"].split_row(
        Layout(name="left", ratio=3),
        Layout(name="right", ratio=2),
    )
    layout["left"].split_column(
        Layout(Panel(Group(*progress_group_items), title="metrics", border_style="green"), ratio=2),
        Layout(Panel(log_text, title="log tail", border_style="yellow"), ratio=3),
    )
    layout["right"].split_column(
        Layout(Panel(resources, title="resources", border_style="magenta"), ratio=1),
        Layout(Panel(history_group, title="history", border_style="cyan"), ratio=1),
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
) -> int:
    last_render: str | None = None
    return_code: int | None = None
    try:
        while True:
            snapshot = load_json(snapshot_path)
            manifest = load_json(manifest_path)
            log_tail = tail_lines(log_path, tail_limit)
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
            if child is not None and child.poll() is not None:
                break
            time.sleep(max(0.2, poll_seconds))
    finally:
        if child is not None:
            return_code = child.wait()

    final_snapshot = load_json(snapshot_path)
    final_manifest = load_json(manifest_path)
    final_render = render_dashboard(
        snapshot=final_snapshot,
        manifest=final_manifest,
        log_path=log_path,
        log_tail=tail_lines(log_path, tail_limit),
        command=command,
        child=child,
        started_at=started_at,
        return_code=return_code,
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
) -> int:
    console = Console()
    return_code: int | None = None
    with Live(console=console, screen=True, auto_refresh=False) as live:
        try:
            while True:
                state = build_dashboard_state(
                    snapshot=load_json(snapshot_path),
                    manifest=load_json(manifest_path),
                    log_path=log_path,
                    log_tail=tail_lines(log_path, tail_limit),
                    command=command,
                    child=child,
                    started_at=started_at,
                )
                live.update(build_rich_dashboard(state), refresh=True)
                if child is not None and child.poll() is not None:
                    break
                time.sleep(max(0.2, poll_seconds))
        finally:
            if child is not None:
                return_code = child.wait()

        final_state = build_dashboard_state(
            snapshot=load_json(snapshot_path),
            manifest=load_json(manifest_path),
            log_path=log_path,
            log_tail=tail_lines(log_path, tail_limit),
            command=command,
            child=child,
            started_at=started_at,
            return_code=return_code,
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
    )


if __name__ == "__main__":
    raise SystemExit(main())
