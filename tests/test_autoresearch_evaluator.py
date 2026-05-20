import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import autoresearch_evaluator as evaluator  # noqa: E402


def _episode(rewards, infos):
    return pl.DataFrame(
        {
            'step': list(range(len(rewards))),
            'reward': rewards,
            'raw_observation': [[0.0, 0.0, 0.0]] * len(rewards),
            'action': [[0.0, 0.0, 0.0]] * len(rewards),
            'info': [json.dumps(info) for info in infos],
        }
    )


def test_summarize_loss_history_reads_final_and_best(tmp_path: Path):
    loss_csv = tmp_path / 'loss.csv'
    with loss_csv.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['epoch', 'train_total', 'train_action', 'train_state', 'train_return', 'val_total', 'val_action', 'val_state', 'val_return'])
        writer.writerow([1, 1.2, 1.0, 0.1, 0.1, 0.8, 0.6, 0.1, 0.1])
        writer.writerow([2, 1.0, 0.8, 0.1, 0.1, 0.6, 0.4, 0.1, 0.1])
        writer.writerow([3, 0.9, 0.7, 0.1, 0.1, 0.7, 0.5, 0.1, 0.1])

    summary = evaluator.summarize_loss_history(loss_csv)

    assert summary['epochs_recorded'] == 3
    assert summary['final_epoch'] == 3
    assert summary['final_val_total_loss'] == pytest.approx(0.7)
    assert summary['best_val_epoch'] == 2
    assert summary['best_val_total_loss'] == pytest.approx(0.6)


def test_compute_info_signal_metrics_counts_violations_and_incidents():
    logs = [
        _episode(
            [1.0, -1.0],
            [
                {'energy_conservation_violation': True, 'battery_soc': 1.0, 'capacity_mwh': 2.0, 'energy_price': 200.0},
                {'deg_incident': True, 'battery_soc': 1.8, 'capacity_mwh': 2.0, 'energy_price': -10.0},
            ],
        )
    ]

    metrics = evaluator.compute_info_signal_metrics(logs)

    assert metrics['episodes_evaluated'] == 1
    assert metrics['violation_step_count'] == 1
    assert metrics['violation_episode_rate'] == pytest.approx(1.0)
    assert metrics['deg_incident_episode_rate'] == pytest.approx(1.0)


def test_evaluate_aemo_heldout_writes_metrics_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    scenario_manifest = [
        {
            'label': 'heldout_nsw1',
            'region': 'NSW1',
            'start_date': datetime(2024, 1, 1),
            'end_date': datetime(2024, 1, 2),
        }
    ]
    processed = {'heldout_nsw1': pl.DataFrame({'RRP': [10.0, 20.0]})}
    cache_preflight_calls: list[dict[str, object]] = []

    monkeypatch.setattr(evaluator, 'fetch_and_preprocess_aemo_scenarios', lambda **kwargs: (processed, scenario_manifest))
    monkeypatch.setattr(
        evaluator,
        'preflight_processed_cache_paths',
        lambda **kwargs: cache_preflight_calls.append(kwargs) or [{'label': 'heldout_nsw1', 'cache_exists': False}],
    )
    monkeypatch.setattr(
        evaluator,
        'resolve_battery_variants',
        lambda variants: [
            {
                'label': 'medium',
                'battery_capacity': 10.0,
                'max_battery_flow': 5.0,
                'init_soc': 5.0,
                'battery_life_cost': 1000.0,
            }
        ],
    )

    def fake_run_policy_episodes(**kwargs):
        policy_name = kwargs['policy_cfg']['name']
        if policy_name == 'candidate_dt':
            return [_episode([3.0, 2.0], [{'battery_soc': 5.0, 'capacity_mwh': 10.0, 'energy_price': 200.0}, {'battery_soc': 4.0, 'capacity_mwh': 10.0, 'energy_price': 100.0}])]
        return [_episode([1.0, 1.0], [{'battery_soc': 5.0, 'capacity_mwh': 10.0, 'energy_price': 10.0}, {'battery_soc': 6.0, 'capacity_mwh': 10.0, 'energy_price': -5.0, 'energy_conservation_violation': True}])]

    monkeypatch.setattr(evaluator, 'run_policy_episodes', fake_run_policy_episodes)

    evaluation_config = {
        'track': 'aemo',
        'target_return': 0.0,
        'bootstrap_iterations': 20,
        'bootstrap_seed': 1,
        'reference_policy': 'rule',
        'heldout': {
            'step_duration': 0.5,
            'episode_hours': 1.0,
            'fit_global_stats': False,
            'battery_variants': [{'name': 'medium'}],
            'scenarios': [
                {
                    'label': 'heldout_nsw1',
                    'region': 'NSW1',
                    'start_date': '2024-01-01',
                    'end_date': '2024-01-02',
                }
            ],
        },
        'policies': [
            {'name': 'candidate_dt', 'kind': 'dt', 'rtg_value': 0.0},
            {'name': 'rule', 'kind': 'rule'},
        ],
    }

    summary = evaluator.evaluate_aemo_heldout(
        surface_manifest={'paths': {}},
        training_summary={'best_val_total_loss': 0.4},
        evaluation_config=evaluation_config,
        output_dir=tmp_path,
        dt_model=object(),
    )

    assert (tmp_path / 'heldout_metrics.csv').exists()
    assert (tmp_path / 'heldout_metrics_by_scenario.csv').exists()
    assert (tmp_path / 'heldout_logs' / 'candidate_dt_heldout_logs.parquet').exists()
    assert summary['reference_policy'] == 'rule'
    assert len(summary['aggregate_metrics']) == 2
    assert 'candidate_dt' in summary['paired_comparisons_vs_reference']
    assert summary['cache_preflight'][0]['label'] == 'heldout_nsw1'
    assert cache_preflight_calls[0]['step_duration'] == pytest.approx(0.5)
