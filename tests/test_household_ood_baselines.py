import datetime as dt

import polars as pl

from scripts.evaluate_household_ood_baselines import (
    _bounded_windows,
    _cache_load,
    _cache_path,
    _cache_store,
    _degradation_cost_from_logs,
    _duration_days,
    _oracle_bill,
    _subsample_windows,
    parse_args,
)


def _segment(days: int, start: dt.datetime = dt.datetime(2026, 1, 1)) -> pl.DataFrame:
    timestamps = [start + dt.timedelta(minutes=5 * index) for index in range(days * 288)]
    return pl.DataFrame({"Timestamp": timestamps, "value": range(len(timestamps))})


def test_bounded_windows_are_complete_evenly_spaced_days():
    windows, provenance = _bounded_windows([_segment(30)], window_days=7, windows_per_segment=3)

    assert len(windows) == 3
    assert all(len(window) == 7 * 288 for window in windows)
    assert [window["Timestamp"][0].day for window in windows] == [1, 13, 24]
    assert all(item["days"] == 7 for item in provenance)


def test_bounded_windows_skip_segments_shorter_than_requested_window():
    windows, provenance = _bounded_windows(
        [_segment(3), _segment(8, dt.datetime(2026, 2, 1))],
        window_days=7,
        windows_per_segment=2,
    )

    assert len(windows) == 2
    assert all(item["source_segment"] == 1 for item in provenance)


def test_bounded_windows_and_duration_support_fifteen_minute_data():
    timestamps = [
        dt.datetime(2026, 1, 1) + dt.timedelta(minutes=15 * index)
        for index in range(10 * 96)
    ]
    segment = pl.DataFrame({"Timestamp": timestamps, "value": range(len(timestamps))})

    windows, provenance = _bounded_windows([segment], window_days=7, windows_per_segment=1)

    assert len(windows[0]) == 7 * 96
    assert _duration_days(windows[0]) == 7.0
    assert provenance[0]["days"] == 7


def test_fixed_standard_rtg_prompt_is_configurable(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate_household_ood_baselines.py", "--dt-rtg-value", "-5"],
    )

    assert parse_args().dt_rtg_value == -5.0


def test_subsample_windows_keeps_evenly_spaced_aligned_records():
    segments = [_segment(1) for _ in range(10)]
    provenance = [{"source_segment": index} for index in range(10)]
    batteries = [{"capacity_kwh": float(index)} for index in range(10)]

    kept, kept_prov, kept_batt = _subsample_windows(segments, provenance, batteries, 4)

    assert [item["source_segment"] for item in kept_prov] == [0, 3, 6, 9]
    assert kept_batt == [batteries[i] for i in (0, 3, 6, 9)]
    assert len(kept) == 4

    same = _subsample_windows(segments, provenance, batteries, None)
    assert same[1] == provenance


def test_synth_surface_flags_are_configurable(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_household_ood_baselines.py",
            "--synth-dir", "data/household/synth_h4_1",
            "--synth-split", "test",
            "--limit-windows", "20",
        ],
    )
    args = parse_args()

    assert str(args.synth_dir).endswith("synth_h4_1")
    assert args.synth_split == "test"
    assert args.limit_windows == 20


def test_workers_and_batch_eval_flags_are_configurable(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_household_ood_baselines.py",
            "--workers", "6",
            "--batch-eval",
        ],
    )
    args = parse_args()
    assert args.workers == 6
    assert args.batch_eval is True


def test_oracle_roundtrip_eff_defaults_to_env_efficiency(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["evaluate_household_ood_baselines.py"],
    )
    assert parse_args().oracle_roundtrip_eff == 0.80

    monkeypatch.setattr(
        "sys.argv",
        ["evaluate_household_ood_baselines.py", "--oracle-roundtrip-eff", "1.0"],
    )
    assert parse_args().oracle_roundtrip_eff == 1.0


def test_reference_cache_dir_is_configurable(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate_household_ood_baselines.py", "--reference-cache-dir", "/tmp/ref_cache"],
    )
    assert str(parse_args().reference_cache_dir).endswith("ref_cache")


def test_reference_cache_roundtrips_and_keys_on_payload(tmp_path):
    payload = {"tariff": "tou", "windows": [{"source_segment": 0}]}
    path = _cache_path(tmp_path, "rule_oracle", payload)
    assert _cache_load(path) is None

    _cache_store(path, ([1.0], [0.0], [{}], [2.0]))
    assert _cache_load(path) == ([1.0], [0.0], [{}], [2.0])

    changed = _cache_path(tmp_path, "rule_oracle", {**payload, "tariff": "flat"})
    assert changed != path
    assert _cache_load(changed) is None


def test_reference_cache_is_a_noop_without_dir():
    assert _cache_path(None, "rule_oracle", {"a": 1}) is None
    _cache_store(None, "ignored")
    assert _cache_load(None) is None


def test_oracle_bill_threads_roundtrip_efficiency(monkeypatch):
    import scripts.evaluate_household_ood_baselines as module

    captured: dict[str, float] = {}

    class _Result:
        bill_aud = 1.0

    def fake_optimize(day, *, tariff, capacity_kwh, max_flow_kw, roundtrip_eff=1.0):
        captured["rte"] = roundtrip_eff
        return _Result()

    monkeypatch.setattr(module, "optimize_dispatch", fake_optimize)
    frame = pl.DataFrame({
        "Timestamp": [
            dt.datetime(2026, 1, 1) + dt.timedelta(minutes=5 * index)
            for index in range(288)
        ],
        "HouseLoad": [0.1] * 288,
        "SolarGen": [0.0] * 288,
    })

    _oracle_bill(frame, 10.0, 5.0, object(), 0.80)
    assert captured["rte"] == 0.80


def test_degradation_cost_uses_step_degradation_and_battery_cost():
    logs = pl.DataFrame({
        "info": [
            {"step_degradation": 0.001},
            {"step_degradation": 0.002},
            {"step_degradation": 0.0},
        ],
    })
    assert _degradation_cost_from_logs(logs, 5000.0) == 15.0
