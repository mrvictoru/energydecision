#!/usr/bin/env python3
"""Build the reproducible H1.5 synthetic household corpus from portal days."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from household_synthetic import (  # noqa: E402
    ARCHETYPES,
    ARCHETYPE_RECIPES,
    BATTERY_CAPACITIES_KWH,
    BATTERY_FLOWS_KW,
    DAY_TYPES,
    ApplianceRecipe,
    DayLibrary,
    TTMConfig,
    apply_ttm_auxiliary,
    assemble_episode,
    day_type_for_date,
    episode_seed,
    inject_appliances,
    scale_solar,
    season_for_date,
    select_ood_source_dates,
    validate_gates,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized-dir", type=Path, default=ROOT / "data/household/real/normalized")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/household/synth")
    parser.add_argument("--episodes", type=int, default=1200)
    parser.add_argument("--days-per-episode", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-clusters", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--ood-holdout-fraction", type=float, default=0.15)
    parser.add_argument("--use-ttm", action="store_true", help="Enable separately provisioned auxiliary Granite TTM.")
    parser.add_argument(
        "--ttm-mode",
        choices=("gap_imputation", "weather_residual"),
        default="weather_residual",
        help="Auxiliary TTM use; never a primary household generator.",
    )
    return parser.parse_args()


def _split_for_index(index_within_archetype: int, per_archetype: int) -> str:
    train_end = int(per_archetype * 0.70)
    val_end = train_end + int(per_archetype * 0.15)
    if index_within_archetype < train_end:
        return "train"
    if index_within_archetype < val_end:
        return "val"
    return "test"


def _job_specs(episodes: int) -> list[tuple[str, str, float, int]]:
    """Distribute jobs round-robin over the 5 x 4 x 3 target surface."""
    specs: list[tuple[str, str, float, int]] = []
    replicate = 0
    while len(specs) < episodes:
        for archetype in ARCHETYPES:
            for season in ("summer", "autumn", "winter", "spring"):
                for capacity in (5.0, 10.0, 20.0):
                    if len(specs) == episodes:
                        return specs
                    specs.append((archetype, season, capacity, replicate))
        replicate += 1
    return specs


def _start_date(library: DayLibrary, season: str, rng: np.random.Generator, days: int) -> dt.date:
    candidates = [
        record.source_date
        for day_type in DAY_TYPES
        for record in library.group_days(season, day_type)
        if all(season_for_date(record.source_date + dt.timedelta(days=index)) == season for index in range(days))
    ]
    if not candidates:
        raise ValueError(f"No {days}-day contiguous source-date anchors available for {season}")
    return candidates[int(rng.integers(len(candidates)))]


def _build_one(
    library: DayLibrary,
    *,
    archetype: str,
    season: str,
    capacity_kwh: float,
    replicate: int,
    master_seed: int,
    days_per_episode: int,
    max_attempts: int,
) -> tuple[object, dict[str, object]]:
    seed = episode_seed(master_seed, archetype, season, capacity_kwh, replicate)
    rng = np.random.default_rng(seed)
    installed_kw = float(rng.uniform(3.0, 15.0))
    orientation_derate = float(rng.uniform(0.75, 1.0))
    flow_kw = float(rng.choice(BATTERY_FLOWS_KW))

    for attempt in range(1, max_attempts + 1):
        # Regenerate sampling/injection parameters after a gate rejection. This
        # follows the H1.5 rule that failed days are regenerated, never edited.
        load_scale = float(rng.uniform(0.4, 3.0))
        intensity = 0.72 ** (attempt - 1)
        base_recipe = ARCHETYPE_RECIPES[archetype]
        recipe = replace(
            base_recipe,
            ev_probability=base_recipe.ev_probability * intensity,
            ac_probability=base_recipe.ac_probability * intensity,
            pool_probability=base_recipe.pool_probability * intensity,
        )
        start = _start_date(library, season, rng, days_per_episode)
        sampled = []
        for offset in range(days_per_episode):
            target = start + dt.timedelta(days=offset)
            sampled.extend(
                library.sample(
                    season_for_date(target),
                    day_type_for_date(target),
                    archetype,
                    n_days=1,
                    scale=load_scale,
                    rng=rng,
                )
            )
        synthetic_days = []
        reference_days = []
        appliance_params = []
        for selected in sampled:
            reference = scale_solar(
                selected.frame,
                installed_kw=installed_kw,
                orientation_derate=orientation_derate,
                aunt_solar_kw=library.aunt_solar_kw,
            )
            generated, appliance = inject_appliances(
                reference,
                season=selected.season,
                archetype=archetype,
                day_type=selected.day_type,
                rng=rng,
                recipe=recipe,
            )
            synthetic_days.append(generated)
            reference_days.append(reference)
            appliance_params.append(appliance)
        gates = validate_gates(synthetic_days, reference_days)
        if gates.passed:
            episode = assemble_episode(synthetic_days, episode_start=start)
            params: dict[str, object] = {
                "seed": seed,
                "attempt": attempt,
                "archetype": archetype,
                "season": season,
                "load_scale_lambda": load_scale,
                "appliance_intensity": intensity,
                "appliance_recipe": asdict(recipe),
                "solar": {
                    "installed_kw": installed_kw,
                    "orientation_derate": orientation_derate,
                    "aunt_solar_kw": library.aunt_solar_kw,
                },
                "battery": {"capacity_kwh": capacity_kwh, "max_flow_kw": flow_kw},
                "source_days": [item.manifest_params() for item in sampled],
                "cluster_weights": {
                    f"{item.season}/{item.day_type}": library.archetype_cluster_weights(
                        item.season, item.day_type, archetype
                    )
                    for item in sampled
                },
                "appliance_days": appliance_params,
                "validation": {"gates": gates.gates, "metrics": gates.metrics},
                "synthetic": True,
                "ttm_used": False,
            }
            return episode, params
    raise RuntimeError(
        f"Could not generate a valid {archetype}/{season} episode after {max_attempts} attempts; "
        f"last gate failures: {gates.failures}"
    )


def main() -> None:
    args = _parse_args()
    if args.episodes < 1 or args.days_per_episode < 1 or args.max_attempts < 1:
        raise ValueError("--episodes, --days-per-episode, and --max-attempts must be positive")

    ttm = TTMConfig(enabled=args.use_ttm, mode=args.ttm_mode if args.use_ttm else "none")
    if ttm.enabled:
        # This fails explicitly until the optional model runtime is provisioned;
        # the corpus builder must never silently substitute a primary generator.
        apply_ttm_auxiliary(None, ttm)  # type: ignore[arg-type]

    full_library = DayLibrary.from_normalized_dir(
        args.normalized_dir, n_clusters=args.n_clusters, random_seed=args.seed
    )
    ood_dates = select_ood_source_dates(full_library, args.ood_holdout_fraction, args.seed)
    library = full_library.excluding_dates(ood_dates)
    specs = _job_specs(args.episodes)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    by_archetype: dict[str, int] = {archetype: 0 for archetype in ARCHETYPES}
    per_archetype = {
        archetype: sum(1 for job_archetype, _, _, _ in specs if job_archetype == archetype)
        for archetype in ARCHETYPES
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "generator": "household_synthetic statistical_recomposition",
        "normalized_dir": str(args.normalized_dir),
        "corpus_params": {
            "episodes": args.episodes,
            "days_per_episode": args.days_per_episode,
            "seed": args.seed,
            "n_clusters": args.n_clusters,
            "max_attempts": args.max_attempts,
            "load_scale_range": [0.4, 3.0],
            "solar_installed_kw_range": [3.0, 15.0],
            "orientation_derate_range": [0.75, 1.0],
            "injection_energy_cap_fraction": 0.60,
            "battery_capacities_kwh": list(BATTERY_CAPACITIES_KWH),
            "battery_flows_kw": list(BATTERY_FLOWS_KW),
            "archetype_recipes": {name: asdict(recipe) for name, recipe in ARCHETYPE_RECIPES.items()},
        },
        "ttm": {"enabled": ttm.enabled, "model_id": ttm.model_id, "mode": ttm.mode},
        "real_ood_source_dates": sorted(date.isoformat() for date in ood_dates),
        "episodes": [],
    }

    for episode_id, (archetype, season, capacity, replicate) in enumerate(specs):
        episode, params = _build_one(
            library,
            archetype=archetype,
            season=season,
            capacity_kwh=capacity,
            replicate=replicate,
            master_seed=args.seed,
            days_per_episode=args.days_per_episode,
            max_attempts=args.max_attempts,
        )
        split = _split_for_index(by_archetype[archetype], per_archetype[archetype])
        by_archetype[archetype] += 1
        file_name = f"{params['seed']}_ep{episode_id}.parquet"
        relative = Path(archetype) / file_name
        target = args.output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        episode.write_parquet(target)
        manifest["episodes"].append({  # type: ignore[index]
            "episode_id": episode_id,
            "path": str(relative),
            "split": split,
            **params,
        })

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.episodes} validated episodes and manifest to {args.output_dir}")


if __name__ == "__main__":
    main()
