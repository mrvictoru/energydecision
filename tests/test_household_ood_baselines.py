import datetime as dt

import polars as pl

from scripts.evaluate_household_ood_baselines import _bounded_windows, _duration_days


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
