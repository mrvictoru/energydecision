"""Traceable, whole-day synthetic household generation for FUTURE_PLAN H1.5.

The primary generator is statistical recomposition: it resamples real,
normalized five-minute days and only adds explicit appliance blocks.  It never
adds row-wise noise, so the source day's intra-day autocorrelation is retained.
All input/load values in this module are kW; :func:`assemble_episode` performs
the kW-to-kWh conversion required by ``SolarBatteryEnv``.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import polars as pl
from scipy.stats import ks_2samp


STEPS_PER_DAY = 288
STEP_HOURS = 5.0 / 60.0
SEASONS = ("summer", "autumn", "winter", "spring")
DAY_TYPES = ("weekday", "weekend")
ARCHETYPES = ("retiree-low", "family-ev", "ac-heavy", "wfh-daytime", "shift-worker")
BATTERY_CAPACITIES_KWH = (5.0, 7.0, 10.0, 13.5, 20.0)
BATTERY_FLOWS_KW = (3.3, 5.0, 7.0)


def season_for_date(value: dt.date | dt.datetime) -> str:
    """Return the Australian meteorological season for ``value``."""
    month = value.month
    if month in (12, 1, 2):
        return "summer"
    if month in (3, 4, 5):
        return "autumn"
    if month in (6, 7, 8):
        return "winter"
    return "spring"


def day_type_for_date(value: dt.date | dt.datetime) -> str:
    return "weekend" if value.weekday() >= 5 else "weekday"


def _daily_energy_kw(profile: np.ndarray) -> float:
    return float(np.sum(profile) * STEP_HOURS)


def _as_numpy(frame: pl.DataFrame, column: str) -> np.ndarray:
    return frame[column].to_numpy().astype(float, copy=False)


def _normalise_profile(profile: np.ndarray) -> np.ndarray:
    energy = float(profile.sum())
    if energy <= 0 or not np.isfinite(energy):
        raise ValueError("A source day must have positive, finite HouseLoad energy")
    return profile / energy


@dataclass(frozen=True)
class ApplianceRecipe:
    """Stochastic appliance controls; values are persisted in each manifest."""

    ev_probability: float = 0.0
    ev_energy_range_kwh: tuple[float, float] = (7.0, 14.0)
    ev_power_kw: float = 7.0
    ac_probability: float = 0.0
    ac_power_range_kw: tuple[float, float] = (2.0, 5.0)
    pool_probability: float = 0.0
    pool_power_kw: float = 1.0
    pool_duration_range_hours: tuple[float, float] = (2.0, 6.0)


ARCHETYPE_RECIPES: dict[str, ApplianceRecipe] = {
    "retiree-low": ApplianceRecipe(pool_probability=0.10),
    "family-ev": ApplianceRecipe(ev_probability=0.80, pool_probability=0.10),
    "ac-heavy": ApplianceRecipe(ac_probability=0.85, pool_probability=0.20),
    "wfh-daytime": ApplianceRecipe(ac_probability=0.20),
    "shift-worker": ApplianceRecipe(ac_probability=0.10),
}


@dataclass
class DayRecord:
    """One complete source day in normalized portal units (kW)."""

    frame: pl.DataFrame
    source_date: dt.date
    season: str
    day_type: str
    load_profile: np.ndarray
    cluster: int = -1

    @property
    def load_energy_kwh(self) -> float:
        return _daily_energy_kw(_as_numpy(self.frame, "HouseLoad"))


@dataclass
class SampledDay:
    """A selected and household-size-scaled source day."""

    frame: pl.DataFrame
    source_date: dt.date
    season: str
    day_type: str
    cluster: int
    archetype: str
    scale: float

    def manifest_params(self) -> dict[str, Any]:
        return {
            "source_date": self.source_date.isoformat(),
            "season": self.season,
            "day_type": self.day_type,
            "cluster": self.cluster,
            "archetype": self.archetype,
            "load_scale_lambda": self.scale,
        }


@dataclass
class GateResult:
    """Results for the automated H1.5 acceptance gates."""

    passed: bool
    gates: dict[str, bool]
    metrics: dict[str, float | int | list[float]]
    failures: list[str] = field(default_factory=list)

    def raise_if_failed(self) -> None:
        if not self.passed:
            raise ValueError("Synthetic validation failed: " + "; ".join(self.failures))


def _kmeans_profiles(profiles: np.ndarray, n_clusters: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Small dependency-free k-means implementation for 288-point profiles."""
    if len(profiles) < 1:
        raise ValueError("Cannot cluster an empty profile set")
    k = min(n_clusters, len(profiles))
    # k-means++ initialization avoids a dependency on scikit-learn.
    centers = [profiles[int(rng.integers(len(profiles)))]]
    while len(centers) < k:
        dist_sq = np.min(
            np.stack([np.sum((profiles - center) ** 2, axis=1) for center in centers]),
            axis=0,
        )
        if dist_sq.sum() <= 0:
            centers.append(profiles[int(rng.integers(len(profiles)))])
        else:
            centers.append(profiles[int(rng.choice(len(profiles), p=dist_sq / dist_sq.sum()))])
    centers_array = np.asarray(centers)

    labels = np.zeros(len(profiles), dtype=int)
    for _ in range(100):
        distances = np.sum((profiles[:, None, :] - centers_array[None, :, :]) ** 2, axis=2)
        updated_labels = np.argmin(distances, axis=1)
        updated_centers = centers_array.copy()
        for index in range(k):
            members = profiles[updated_labels == index]
            if len(members):
                updated_centers[index] = members.mean(axis=0)
        if np.array_equal(labels, updated_labels) and np.allclose(centers_array, updated_centers):
            labels = updated_labels
            centers_array = updated_centers
            break
        labels = updated_labels
        centers_array = updated_centers
    return labels, centers_array


def cluster_purity(cluster_labels: Sequence[int], held_out_labels: Sequence[Any]) -> float:
    """Return majority-label purity, useful with externally supplied test labels."""
    if len(cluster_labels) != len(held_out_labels) or len(cluster_labels) == 0:
        raise ValueError("cluster_labels and held_out_labels must be non-empty and equal length")
    labels = np.asarray(cluster_labels)
    targets = np.asarray(held_out_labels)
    correct = 0
    for cluster in np.unique(labels):
        members = targets[labels == cluster]
        _, counts = np.unique(members, return_counts=True)
        correct += int(counts.max())
    return correct / len(labels)


class DayLibrary:
    """Clustered whole-day library, grouped by season and weekday/weekend."""

    def __init__(
        self,
        days: Sequence[DayRecord],
        n_clusters: int = 4,
        random_seed: int = 0,
    ) -> None:
        if not days:
            raise ValueError("DayLibrary requires at least one complete source day")
        if n_clusters < 1:
            raise ValueError("n_clusters must be positive")
        self.days = list(days)
        self.n_clusters = n_clusters
        self.random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)
        self._groups: dict[tuple[str, str], list[DayRecord]] = {}
        self._centers: dict[tuple[str, str], np.ndarray] = {}
        self._fit()
        solar_peaks = np.concatenate([_as_numpy(day.frame, "SolarGen") for day in self.days])
        self.aunt_solar_kw = float(np.quantile(solar_peaks, 0.995))
        if self.aunt_solar_kw <= 0:
            raise ValueError("Source corpus has no positive solar generation for solar scaling")

    @classmethod
    def from_normalized_dir(
        cls,
        normalized_dir: str | Path,
        n_clusters: int = 4,
        random_seed: int = 0,
        excluded_dates: Iterable[dt.date] = (),
    ) -> "DayLibrary":
        """Load only complete, contiguous 288-row normalized portal days."""
        excluded = set(excluded_dates)
        files = sorted(Path(normalized_dir).glob("*_normalized.parquet"))
        if not files:
            raise FileNotFoundError(f"No *_normalized.parquet under {normalized_dir}")
        days: list[DayRecord] = []
        for path in files:
            frame = pl.read_parquet(path).sort("Timestamp")
            frame = frame.with_columns(pl.col("Timestamp").dt.date().alias("_source_date"))
            for day_frame in frame.partition_by("_source_date", maintain_order=True):
                day_frame = day_frame.drop("_source_date")
                timestamps = day_frame["Timestamp"].to_list()
                if len(day_frame) != STEPS_PER_DAY or not timestamps:
                    continue
                source_date = timestamps[0].date()
                if source_date in excluded:
                    continue
                deltas = np.diff(day_frame["Timestamp"].cast(pl.Int64).to_numpy())
                if len(deltas) and not np.all(deltas == 5 * 60 * 1_000_000):
                    continue
                required = {"Timestamp", "HouseLoad", "SolarGen", "ImportEnergyPrice", "ExportEnergyPrice"}
                if not required.issubset(day_frame.columns):
                    raise ValueError(f"{path} is missing normalized columns {sorted(required - set(day_frame.columns))}")
                load = _as_numpy(day_frame, "HouseLoad")
                solar = _as_numpy(day_frame, "SolarGen")
                if not np.isfinite(load).all() or not np.isfinite(solar).all() or np.any(load < 0) or np.any(solar < 0):
                    continue
                # G5 is an acceptance gate, so reject a bad source day rather
                # than repairing a zero reading after it has been resampled.
                if _daily_energy_kw(load) <= 0 or np.any((load <= 1e-12) & (solar <= 1e-12)):
                    continue
                days.append(
                    DayRecord(
                        frame=day_frame,
                        source_date=source_date,
                        season=season_for_date(source_date),
                        day_type=day_type_for_date(source_date),
                        load_profile=_normalise_profile(load),
                    )
                )
        if not days:
            raise ValueError("No complete valid normalized days remained after filtering")
        return cls(days, n_clusters=n_clusters, random_seed=random_seed)

    def excluding_dates(self, excluded_dates: Iterable[dt.date]) -> "DayLibrary":
        excluded = set(excluded_dates)
        return DayLibrary(
            [day for day in self.days if day.source_date not in excluded],
            n_clusters=self.n_clusters,
            random_seed=self.random_seed,
        )

    def _fit(self) -> None:
        for season in SEASONS:
            for day_type in DAY_TYPES:
                key = (season, day_type)
                members = [day for day in self.days if day.season == season and day.day_type == day_type]
                if not members:
                    continue
                profiles = np.stack([day.load_profile for day in members])
                labels, centers = _kmeans_profiles(profiles, self.n_clusters, self._rng)
                for day, label in zip(members, labels):
                    day.cluster = int(label)
                self._groups[key] = members
                self._centers[key] = centers

    @property
    def cluster_centers(self) -> Mapping[tuple[str, str], np.ndarray]:
        return self._centers

    def group_days(self, season: str, day_type: str, cluster: int | None = None) -> list[DayRecord]:
        """Return a copy of matching source-day references."""
        self._validate_group(season, day_type)
        values = self._groups[(season, day_type)]
        if cluster is not None:
            values = [day for day in values if day.cluster == cluster]
        return list(values)

    def _validate_group(self, season: str, day_type: str) -> None:
        if season not in SEASONS:
            raise ValueError(f"Unknown season {season!r}; expected one of {SEASONS}")
        if day_type not in DAY_TYPES:
            raise ValueError(f"Unknown day_type {day_type!r}; expected one of {DAY_TYPES}")
        if (season, day_type) not in self._groups:
            raise ValueError(f"No source days available for {season}/{day_type}")

    def archetype_cluster_weights(self, season: str, day_type: str, archetype: str) -> dict[int, float]:
        """Derive explicit per-cluster recipe weights from normalized profiles."""
        if archetype not in ARCHETYPES:
            raise ValueError(f"Unknown archetype {archetype!r}")
        self._validate_group(season, day_type)
        centers = self._centers[(season, day_type)]
        slots = np.arange(STEPS_PER_DAY) / 12.0
        masks = {
            "early": (slots >= 5) & (slots < 10),
            "day": (slots >= 9) & (slots < 16),
            "afternoon": (slots >= 12) & (slots < 18),
            "evening": (slots >= 16) & (slots < 23),
            "overnight": (slots < 5) | (slots >= 23),
        }
        values: list[float] = []
        for profile in centers:
            mean = float(profile.mean())
            flatness = -float(profile.std() / mean) if mean else 0.0
            periods = {name: float(profile[mask].sum()) for name, mask in masks.items()}
            score = {
                "retiree-low": 1.5 * periods["early"] + 0.5 * periods["day"] + flatness,
                "family-ev": 2.0 * periods["evening"] + 0.5 * periods["overnight"],
                "ac-heavy": 2.0 * periods["afternoon"] + periods["day"],
                "wfh-daytime": 2.0 * periods["day"] + periods["afternoon"],
                "shift-worker": 2.0 * periods["overnight"] - periods["evening"],
            }[archetype]
            values.append(score)
        logits = np.asarray(values) * 8.0
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        return {index: float(prob) for index, prob in enumerate(probs)}

    def sample(
        self,
        season: str,
        day_type: str,
        archetype: str,
        n_days: int,
        scale: float,
        rng: np.random.Generator | None = None,
    ) -> list[SampledDay]:
        """Bootstrap complete matching days and scale only the load by ``scale``."""
        if n_days < 1:
            raise ValueError("n_days must be positive")
        if not 0.4 <= scale <= 3.0:
            raise ValueError("scale must be in the H1.5 range [0.4, 3.0]")
        if archetype not in ARCHETYPES:
            raise ValueError(f"Unknown archetype {archetype!r}")
        rng = rng or self._rng
        weights = self.archetype_cluster_weights(season, day_type, archetype)
        clusters = np.asarray(sorted(weights))
        probabilities = np.asarray([weights[cluster] for cluster in clusters])
        selected: list[SampledDay] = []
        for _ in range(n_days):
            cluster = int(rng.choice(clusters, p=probabilities))
            members = self.group_days(season, day_type, cluster)
            if not members:
                members = self.group_days(season, day_type)
            source = members[int(rng.integers(len(members)))]
            frame = source.frame.with_columns((pl.col("HouseLoad") * scale).alias("HouseLoad"))
            selected.append(
                SampledDay(frame, source.source_date, season, day_type, source.cluster, archetype, scale)
            )
        return selected


def inject_appliances(
    frame: pl.DataFrame,
    *,
    season: str,
    archetype: str,
    day_type: str,
    rng: np.random.Generator,
    recipe: ApplianceRecipe | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Add explicit EV, AC, and pool blocks, capped at 60% of daily load energy."""
    if len(frame) != STEPS_PER_DAY:
        raise ValueError("Appliance injection requires exactly one complete day")
    if archetype not in ARCHETYPE_RECIPES:
        raise ValueError(f"Unknown archetype {archetype!r}")
    recipe = recipe or ARCHETYPE_RECIPES[archetype]
    injection = np.zeros(STEPS_PER_DAY, dtype=float)
    params: dict[str, Any] = {
        "ev_active": False,
        "ac_active": False,
        "pool_active": False,
        "injection_cap_fraction": 0.60,
    }

    # EV: truncated N(18h, 45 min), only households whose recipe opts in.
    ev_probability = recipe.ev_probability
    if archetype == "family-ev" and day_type == "weekend":
        ev_probability = min(1.0, ev_probability + 0.10)
    if rng.random() < ev_probability:
        start_hour = float(np.clip(rng.normal(18.0, 0.75), 16.0, 22.0))
        energy = float(rng.uniform(*recipe.ev_energy_range_kwh))
        duration_hours = energy / recipe.ev_power_kw
        start = int(round(start_hour * 12))
        end = min(STEPS_PER_DAY, start + max(1, int(math.ceil(duration_hours * 12))))
        injection[start:end] += recipe.ev_power_kw
        params.update({
            "ev_active": True,
            "ev_start_hour": start_hour,
            "ev_energy_requested_kwh": energy,
            "ev_power_kw": recipe.ev_power_kw,
        })

    # AC uses a two-state hourly Markov duty cycle, conditioned on warm
    # afternoon hours. Hourly blocks preserve realistic five-minute ramps.
    if season == "summer" and rng.random() < recipe.ac_probability:
        ac_power = float(rng.uniform(*recipe.ac_power_range_kw))
        on = False
        transitions = 0
        for hour in range(12, 17):
            p_on = 0.38 if 13 <= hour <= 16 else 0.15
            p_off = 0.10 if 13 <= hour <= 16 else 0.25
            was_on = on
            if on:
                on = rng.random() >= p_off
            else:
                on = rng.random() < p_on
            if on:
                injection[hour * 12:(hour + 1) * 12] += ac_power
            transitions += int(on != was_on)
        params.update({"ac_active": True, "ac_power_kw": ac_power, "ac_transitions": transitions})

    if rng.random() < recipe.pool_probability:
        duration = float(rng.uniform(*recipe.pool_duration_range_hours))
        start_hour = float(rng.uniform(9.0, max(9.0, 17.0 - duration)))
        start = int(round(start_hour * 12))
        end = min(STEPS_PER_DAY, start + max(1, int(round(duration * 12))))
        injection[start:end] += recipe.pool_power_kw
        params.update({
            "pool_active": True,
            "pool_start_hour": start_hour,
            "pool_duration_hours": duration,
            "pool_power_kw": recipe.pool_power_kw,
        })

    base_energy = _daily_energy_kw(_as_numpy(frame, "HouseLoad"))
    requested_energy = _daily_energy_kw(injection)
    cap_energy = base_energy * 0.60
    applied_scale = min(1.0, cap_energy / requested_energy) if requested_energy > 0 else 1.0
    injection *= applied_scale
    params.update({
        "base_load_energy_kwh": base_energy,
        "injection_requested_kwh": requested_energy,
        "injection_applied_kwh": _daily_energy_kw(injection),
        "injection_scale": applied_scale,
    })
    return frame.with_columns((pl.col("HouseLoad") + pl.Series(injection)).alias("HouseLoad")), params


def scale_solar(
    frame: pl.DataFrame,
    *,
    installed_kw: float,
    orientation_derate: float,
    aunt_solar_kw: float,
) -> pl.DataFrame:
    """Scale a real solar curve by roof size and orientation without reshaping it."""
    if not 3.0 <= installed_kw <= 15.0:
        raise ValueError("installed_kw must be in the H1.5 range [3, 15]")
    if not 0.75 <= orientation_derate <= 1.0:
        raise ValueError("orientation_derate must be in the H1.5 range [0.75, 1.0]")
    if aunt_solar_kw <= 0:
        raise ValueError("aunt_solar_kw must be positive")
    factor = installed_kw / aunt_solar_kw * orientation_derate
    return frame.with_columns((pl.col("SolarGen") * factor).alias("SolarGen"))


def _autocorrelations(values: np.ndarray, max_lag: int = 12) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    output = []
    for lag in range(1, max_lag + 1):
        left, right = values[:-lag], values[lag:]
        if left.std() < 1e-12 or right.std() < 1e-12:
            output.append(1.0 if np.allclose(left, right) else 0.0)
        else:
            output.append(float(np.corrcoef(left, right)[0, 1]))
    return np.asarray(output)


def _peak_modes(days: Sequence[pl.DataFrame], start_hour: int, end_hour: int) -> int:
    slots = []
    start, end = start_hour * 12, end_hour * 12
    for frame in days:
        load = _as_numpy(frame, "HouseLoad")
        slots.append(start + int(np.argmax(load[start:end])))
    counts = np.bincount(slots, minlength=STEPS_PER_DAY)
    return int(np.argmax(counts))


def validate_gates(
    synthetic_days: Sequence[pl.DataFrame],
    reference_days: Sequence[pl.DataFrame],
    *,
    ks_min_pvalue: float = 0.05,
) -> GateResult:
    """Evaluate G1-G6; callers must regenerate rejected candidates."""
    if not synthetic_days or not reference_days:
        raise ValueError("Validation requires synthetic and matching reference days")
    if any(len(day) != STEPS_PER_DAY for day in (*synthetic_days, *reference_days)):
        raise ValueError("Validation requires complete 288-step days")
    synth_load = [_as_numpy(day, "HouseLoad") for day in synthetic_days]
    ref_load = [_as_numpy(day, "HouseLoad") for day in reference_days]
    synth_solar = [_as_numpy(day, "SolarGen") for day in synthetic_days]
    ref_solar = [_as_numpy(day, "SolarGen") for day in reference_days]

    synth_energy = np.asarray([_daily_energy_kw(day) for day in synth_load])
    ref_energy = np.asarray([_daily_energy_kw(day) for day in ref_load])
    ks = ks_2samp(synth_energy, ref_energy, method="asymp")
    g1 = bool(ks.pvalue > ks_min_pvalue)

    morning_shift = abs(_peak_modes(synthetic_days, 5, 12) - _peak_modes(reference_days, 5, 12)) / 12.0
    evening_shift = abs(_peak_modes(synthetic_days, 16, 23) - _peak_modes(reference_days, 16, 23)) / 12.0
    g2 = bool(morning_shift <= 1.0 and evening_shift <= 1.0)

    synth_ramp = float(np.quantile(np.abs(np.diff(np.concatenate(synth_load))), 0.95))
    ref_ramp = float(np.quantile(np.abs(np.diff(np.concatenate(ref_load))), 0.95))
    ramp_ratio = synth_ramp / ref_ramp if ref_ramp > 1e-9 else (1.0 if synth_ramp <= 1e-9 else math.inf)
    g3 = bool(0.8 <= ramp_ratio <= 1.2)

    synth_acf = _autocorrelations(np.concatenate(synth_load))
    ref_acf = _autocorrelations(np.concatenate(ref_load))
    acf_max_diff = float(np.max(np.abs(synth_acf - ref_acf)))
    g4 = bool(acf_max_diff <= 0.1)

    zero_rows = int(sum(np.sum((load <= 1e-12) & (solar <= 1e-12)) for load, solar in zip(synth_load, synth_solar)))
    g5 = zero_rows == 0

    physical_values = np.concatenate([np.concatenate(synth_load), np.concatenate(synth_solar)])
    g6 = bool(np.isfinite(physical_values).all() and np.all(physical_values >= 0))
    gates = {"G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5, "G6": g6}
    metrics: dict[str, float | int | list[float]] = {
        "g1_ks_pvalue": float(ks.pvalue),
        "g2_morning_mode_shift_hours": morning_shift,
        "g2_evening_mode_shift_hours": evening_shift,
        "g3_synthetic_ramp_p95_kw": synth_ramp,
        "g3_reference_ramp_p95_kw": ref_ramp,
        "g3_ramp_ratio": ramp_ratio,
        "g4_acf_max_abs_diff": acf_max_diff,
        "g4_synthetic_acf": synth_acf.tolist(),
        "g4_reference_acf": ref_acf.tolist(),
        "g5_zero_energy_rows": zero_rows,
    }
    failures = [name for name, passed in gates.items() if not passed]
    return GateResult(not failures, gates, metrics, failures)


def assemble_episode(
    days: Sequence[pl.DataFrame],
    *,
    episode_start: dt.date,
) -> pl.DataFrame:
    """Export a continuous 7-day (or arbitrary whole-day) env-view dataframe."""
    if not days or any(len(day) != STEPS_PER_DAY for day in days):
        raise ValueError("Episode assembly requires one or more complete days")
    frames = []
    for day_index, source in enumerate(days):
        start = dt.datetime.combine(episode_start + dt.timedelta(days=day_index), dt.time())
        timestamps = [start + dt.timedelta(minutes=5 * slot) for slot in range(STEPS_PER_DAY)]
        frames.append(
            source.with_columns([
                pl.Series("Timestamp", timestamps),
                pl.Series("Time", timestamps),
                (pl.col("HouseLoad") * STEP_HOURS).alias("HouseLoad"),
                (pl.col("SolarGen") * STEP_HOURS).alias("SolarGen"),
            ])
        )
    combined = pl.concat(frames).select([
        "Timestamp", "Time", "SolarGen", "HouseLoad", "ImportEnergyPrice", "ExportEnergyPrice",
    ])
    combined = combined.with_columns([
        pl.col("SolarGen").shift(STEPS_PER_DAY).fill_null(pl.col("SolarGen")).alias("FutureSolar"),
        pl.col("HouseLoad").shift(STEPS_PER_DAY).fill_null(pl.col("HouseLoad")).alias("FutureLoad"),
    ])
    # Keep the only columns SolarBatteryEnv treats as observations.
    return combined.select([
        "Timestamp", "Time", "SolarGen", "HouseLoad",
        "FutureSolar", "FutureLoad", "ImportEnergyPrice", "ExportEnergyPrice",
    ])


def select_ood_source_dates(library: DayLibrary, fraction: float, seed: int) -> set[dt.date]:
    """Stratify real held-out source days by season/day type before synthesis."""
    if not 0.0 <= fraction < 1.0:
        raise ValueError("fraction must be in [0, 1)")
    rng = np.random.default_rng(seed)
    selected: set[dt.date] = set()
    for members in library._groups.values():
        count = int(round(len(members) * fraction))
        # Preserve at least one source record for every occupied stratum.
        count = min(count, max(0, len(members) - 1))
        if count:
            choices = rng.choice(len(members), size=count, replace=False)
            selected.update(members[int(index)].source_date for index in choices)
    return selected


def episode_seed(master_seed: int, archetype: str, season: str, capacity_kwh: float, replicate: int) -> int:
    """Stable seed independent of Python's randomized ``hash()``."""
    text = f"{master_seed}|{archetype}|{season}|{capacity_kwh}|{replicate}".encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big")


@dataclass(frozen=True)
class TTMConfig:
    """Flag-gated auxiliary TTM configuration; never used as primary generation."""

    enabled: bool = False
    model_id: str = "ibm-granite/granite-timeseries-ttm-r2"
    mode: str = "none"


def apply_ttm_auxiliary(frame: pl.DataFrame, config: TTMConfig) -> pl.DataFrame:
    """Reserve an isolated integration point for opt-in TTM residual/imputation."""
    if not config.enabled:
        return frame
    if config.mode not in {"gap_imputation", "weather_residual"}:
        raise ValueError("TTM mode must be 'gap_imputation' or 'weather_residual' when enabled")
    raise RuntimeError(
        "TTM inference is intentionally isolated from the primary generator and "
        "requires a separately provisioned Granite TTM runtime; no TTM output was produced."
    )
