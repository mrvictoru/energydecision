from __future__ import annotations

import json
from typing import Any


class ParseError(ValueError):
    pass


def build_system_prompt(allowed_keys: frozenset, mutable_params: dict[str, Any]) -> str:
    lines = [
        "You are a hyperparameter mutation proposer.",
        "Respond with ONLY a JSON object containing key:value pairs to change.",
        "Allowed keys:",
    ]
    for key in sorted(allowed_keys):
        meta = mutable_params.get(key, {})
        desc = meta.get("description", "")
        typ = meta.get("type", "any")
        rng = meta.get("range", "")
        extra = f" [{typ}] {rng}".strip()
        lines.append(f"- {key}{(': ' + desc) if desc else ''}{(' ' + extra) if extra else ''}")
    return "\n".join(lines)


def build_user_prompt(
    best_config: dict,
    best_metric: float | str,
    metric_name: str,
    history_lines: str,
    constraint_clause: str = "",
) -> str:
    tail = f" with constraints: {constraint_clause}" if constraint_clause else ""
    return (
        "Current best config:\n"
        f"{json.dumps(best_config, indent=2)}\n\n"
        f"Current best {metric_name}: {best_metric}\n\n"
        "Recent history:\n"
        f"{history_lines or '- none'}\n\n"
        f"Propose a new configuration that improves {metric_name}{tail}."
    )


def _extract_first_json_object(raw: str) -> str:
    start = raw.find("{")
    if start < 0:
        raise ParseError("No JSON object found")

    depth = 0
    in_string = False
    escaped = False

    for idx in range(start, len(raw)):
        ch = raw[idx]
        if in_string:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start : idx + 1]

    raise ParseError("Unbalanced JSON object in LLM response")


def _cast_value(value: Any) -> Any:
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, (list, dict)):
        return value
    return str(value)


def parse_llm_response(raw: str, allowed_keys: frozenset) -> dict[str, Any]:
    block = _extract_first_json_object(raw)
    try:
        payload = json.loads(block)
    except json.JSONDecodeError as exc:
        raise ParseError("Failed to decode JSON from LLM response") from exc

    if not isinstance(payload, dict):
        raise ParseError("LLM JSON must be an object")

    parsed: dict[str, Any] = {}
    for key, value in payload.items():
        if key not in allowed_keys:
            continue
        try:
            parsed[key] = _cast_value(value)
        except Exception:  # noqa: BLE001
            continue

    if not parsed:
        raise ParseError("No valid mutable keys found in LLM response")

    return parsed
