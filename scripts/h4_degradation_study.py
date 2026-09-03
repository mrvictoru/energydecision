#!/usr/bin/env python3
"""
H4.5 Degradation-Aware Policy Study

Compare DT policies trained with different degradation configurations:
1. Degradation disabled (battery_life_cost=0, no capacity fade)
2. Cycle-only degradation (custom DegradationModel with calendar aging disabled)
3. Full realistic degradation (calendar + cycle, battery_life_cost=5000) - default
4. High degradation cost (battery_life_cost=10000)
5. Low degradation cost (battery_life_cost=1000)

Trains and evaluates Decision Transformers on the horizon-diverse household corpus (H4.1).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
from datetime import datetime

import numpy as np
import polars as pl
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from EnergySimEnv import SolarBatteryEnv
from decision_transformer import DecisionTransformer
from transformer_training import (
    train_decision_transformer,
    TrajectoryDataset,
    merge_trajectory_datasets,
    load_trajectory_datasets,
)
from generate_household_sdp_trajectories import main as generate_sdp_trajectories
from generate_household_sdp_trajectories import parse_args as gen_parse_args
from pretrain_decision_transformer import main as train_dt
from pretrain_decision_transformer import parse_args as train_parse_args
from evaluate_household_ood_baselines import main as evaluate_baselines
from evaluate_household_ood_baselines import parse_args as eval_parse_args


@dataclass
class DegradationConfig:
    """Configuration for a degradation scenario."""
    name: str
    battery_life_cost: float
    degradation_mode: str  # "full", "cycle_only", "disabled"
    description: str


DEGRADATION_CONFIGS = [
    {
        "name": "degradation_disabled",
        "battery_life_cost": 0.0,
        "degradation_mode": "disabled",
        "description": "No degradation cost, no capacity fade"
    },
    {
        "name": "cycle_only",
        "battery_life_cost": 5000.0,
        "degradation_mode": "cycle_only",
        "description": "Cycle aging only, no calendar aging"
    },
    {
        "name": "full_realistic",
        "battery_life_cost": 5000.0,
        "degradation_mode": "full",
        "description": "Full realistic: calendar + cycle aging, $5000 battery life cost"
    },
    {
        "name": "high_degradation_cost",
        "battery_life_cost": 10000.0,
        "degradation_mode": "full",
        "description": "High degradation cost ($10k), full realistic"
    },
    {
        "name": "low_degradation_cost",
        "battery_life_cost": 1000.0,
        "degradation_mode": "full",
        "description": "Low degradation cost ($1k), full realistic"
    },
]


def build_corpus_if_needed(corpus_dir: Path) -> Path:
    """Build the H4.1 corpus if it doesn't exist."""
    corpus_dir = Path(corpus_dir)
    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"[H4.5] Building H4.1 corpus at {corpus_dir}...")
        subprocess.run([
            "python3", "scripts/build_household_synth_corpus.py",
            "--output-dir", str(corpus_dir),
            "--episodes", "240",
            "--horizons", "1w", "2w", "6m", "2y",
            "--seed", "20260830"
        ], check=True, cwd=ROOT)
    return corpus_dir


def generate_sdp_trajectories(config: dict, corpus_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    """Generate SDP teacher trajectories for train and val splits."""
    print(f"[H4.5] Generating SDP trajectories for {config['name']}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_out = output_dir / "sdp_teacher_train.parquet"
    val_out = output_dir / "sdp_teacher_val.parquet"
    
    # Generate train split
    subprocess.run([
        "python3", "scripts/generate_household_sdp_trajectories.py",
        "--synth-dir", str(corpus_dir),
        "--split", "train",
        "--out", str(train_out),
        "--degradation-mode", config["degradation_mode"],
        "--battery-life-cost", str(config["battery_life_cost"]),
    ], check=True, cwd=ROOT)
    
    # Generate val split
    subprocess.run([
        "python3", "scripts/generate_household_sdp_trajectories.py",
        "--synth-dir", str(corpus_dir),
        "--split", "val",
        "--out", str(val_out),
        "--degradation-mode", config["degradation_mode"],
        "--battery-life-cost", str(config["battery_life_cost"]),
    ], check=True, cwd=ROOT)
    
    return train_out, val_out


def train_dt_model(config: dict, train_path: Path, val_path: Path, output_dir: Path) -> Path:
    """Train a Decision Transformer model."""
    print(f"[H4.5] Training DT model for {config['name']}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / f"dt_{config['name']}_best.pt"
    
    # Build the training command
    cmd = [
        "python3", "scripts/pretrain_decision_transformer.py",
        "--surface-preset", "household_baseline",
        "--data-dir", str(config["train_dir"]),
        "--patterns", config["train_pattern"],
        "--val-data-dir", str(config["val_dir"]),
        "--val-patterns", config["val_pattern"],
        "--split-policy", "explicit_validation",
        "--context-length", "576",
        "--stride", "288",
        "--n-block", "8",
        "--h-dim", "512",
        "--n-heads", "8",
        "--drop-p", "0.15",
        "--batch-size", "16",
        "--epochs", "5",
        "--lr", "3e-5",
        "--seed", "42",
        "--rtg-source", "constant",
        "--return-scale", "1.0",
        "--action-loss-weight", "0.999",
        "--state-loss-weight", "0.002",
        "--return-loss-weight", "0.0001",
        "--device", "cuda",
        "--amp-mode", "auto",
        "--save-path", str(output_dir / f"dt_{config['name']}.pt"),
        "--checkpoint-path", str(output_dir / f"dt_{config['name']}_checkpoint.pt"),
        "--loss-csv-path", str(output_dir / f"dt_{config['name']}_loss.csv"),
    ]
    
    print(f"[H4.5] Training command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT)
    
    return Path(config["save_path"])


def evaluate_model(model_path: str, config: dict, corpus_dir: Path) -> dict:
    """Evaluate a trained model on the test split."""
    print(f"[H4.5] Evaluating model: {config['name']}")
    
    # Run the household OOD evaluation
    eval_output_dir = Path(f"eval_output/household/h4_5_degradation/{config['name']}")
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    
    subprocess.run([
        "python3", "scripts/evaluate_household_ood_baselines.py",
        "--normalized-dir", "data/household/real/normalized",
        "--output-dir", str(eval_output_dir),
        "--dt-path", config["model_path"],
        "--dt-config", config["model_config_path"],
        "--dt-rtg-mode", "standard",
        "--dt-rtg-value", "-2",
        "--forecast-mode", "persistence",
        "--tariff", "realistic",
        "--window-days", "7",
        "--windows-per-segment", "2",
        "--skip-reference-policies",
        "--skip-ppo",
        "--device", "cuda",
    ], check=True, cwd=ROOT)
    
    # Read results
    summary_path = eval_output_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="H4.5 Degradation-Aware Policy Study")
    parser.add_argument("--config", choices=["degradation_disabled", "cycle_only", "full_realistic", 
                        "high_degradation_cost", "low_degradation_cost", "all"],
                        default="all", help="Degradation config to run")
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument("--eval", action="store_true", help="Evaluate trained models")
    parser.add_argument("--output-dir", type=Path, default=Path("results/h4_5_degradation"))
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/household/synth_h4_1"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "src"))
    
    configs = [
        {"name": "degradation_disabled", "battery_life_cost": 0.0, "degradation_mode": "disabled", 
         "description": "No degradation cost, no capacity fade"},
        {"name": "cycle_only", "battery_life_cost": 5000.0, "degradation_mode": "cycle_only",
         "description": "Cycle aging only, no calendar aging"},
        {"name": "full_realistic", "battery_life_cost": 5000.0, "degradation_mode": "full",
         "description": "Full realistic: calendar + cycle aging, $5000 battery life cost"},
        {"name": "high_degradation_cost", "battery_life_cost": 10000.0, "degradation_mode": "full",
         "description": "High degradation cost ($10k), full realistic"},
        {"name": "low_degradation_cost", "battery_life_cost": 1000.0, "degradation_mode": "full",
         "description": "Low degradation cost ($1k), full realistic"},
    ]
    
    if args.config != "all":
        configs = [c for c in configs if c["name"] == args.config]
        if not configs:
            print(f"Unknown config: {args.config}")
            return 1
    
    # Build corpus if needed
    corpus_dir = Path("data/household/synth_h4_1")
    if not (Path("data/household/synth_h4_1/manifest.json").exists()):
        print("[H4.5] Building H4.1 corpus...")
        import subprocess
        subprocess.run([
            "python3", "scripts/build_household_synth_corpus.py",
            "--output-dir", "data/household/synth_h4_1",
            "--episodes", "240",
            "--horizons", "1w", "2w", "6m", "2y",
            "--seed", "20260830"
        ], check=True, cwd=ROOT)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"[H4.5] Running {config['name']}: {config['description']}")
        print(f"{'='*60}")
        
        # Generate SDP trajectories
        sdp_output_dir = Path(f"data/household/dt/h4_5_{config['name']}")
        train_out, val_out = generate_sdp_trajectories({
            "degradation_mode": config["degradation_mode"],
            "battery_life_cost": config["battery_life_cost"],
        }, Path("data/household/synth_h4_1"), Path(f"data/household/dt/h4_5_{config['name']}"))
        
        # Train DT model
        model_path = train_dt_model(config, Path(config["train_dir"]), Path(config["val_dir"]), Path(config["model_dir"]))
        
        # Evaluate
        eval_results = evaluate_model(model_path, config, Path("data/household/synth_h4_1"))
        
        results[config["name"]] = {
            "config": config,
            "model_path": config["model_path"],
            "eval_results": eval_results
        }
    
    # Save summary
    output_dir = Path("results/h4_5_degradation")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[H4.5] Completed. Results in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())