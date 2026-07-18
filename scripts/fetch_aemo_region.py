"""Fetch and process AEMO data for a single region."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    root = repo_root()
    sys.path.insert(0, str(root / "src"))
    
    from aemo_notebook_utils import fetch_and_preprocess_aemo_data
    
    parser = argparse.ArgumentParser(description="Fetch and cache AEMO processed data for one region")
    parser.add_argument("--region", required=True, help="NEM region (NSW1, QLD1, SA1, TAS1, VIC1)")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Cache directory")
    parser.add_argument("--step-duration", type=float, default=5/60, help="Step duration in hours")
    args = parser.parse_args()
    
    cache_dir = args.cache_dir or root / "data" / "aemo"
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    region = args.region.upper()
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    sd = args.step_duration
    
    print(f"Fetching {region} from {start.date()} to {end.date()}...")
    print(f"Cache: {cache_dir}")
    print(f"Step duration: {sd:.4f}h")
    print(f"Starting at: {datetime.now()}")
    print()
    
    try:
        processed, cache_path = fetch_and_preprocess_aemo_data(
            region=region,
            start_date=start,
            end_date=end,
            cache_dir=cache_dir,
            step_duration=sd,
        )
        print(f"Done! Processed {len(processed)} rows.")
        print(f"Cached at: {cache_path}")
        # Update progress tracker
        progress_path = root / "data_fetch_progress.md"
        if progress_path.exists():
            content = progress_path.read_text()
            replacements = [
                (f"**{region}**", f"**{region}** ✅ COMPLETE"),
            ]
            for old, new in replacements:
                if old in content:
                    content = content.replace(old, new)
            progress_path.write_text(content)
            print(f"Updated {progress_path}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
