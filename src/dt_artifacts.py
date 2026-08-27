"""Small, explicit persistence helpers for Decision Transformer artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def write_model_kwargs(path: str | Path, model_kwargs: Mapping[str, Any]) -> Path:
    """Write the exact construction kwargs next to a DT checkpoint."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(model_kwargs), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output
