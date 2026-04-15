"""Shared helpers for experiment artifacts."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def ensure_parent(path: str | Path) -> Path:
    """Create a file path's parent directory and return the file path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def now_iso() -> str:
    """Return an ISO timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sanitize_name(value: str) -> str:
    """Make a filesystem-friendly name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    cleaned = cleaned.strip(".-")
    return cleaned or "run"


def write_json(path: str | Path, payload: Any) -> Path:
    """Write JSON with a stable layout."""
    out = ensure_parent(path)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


def read_json(path: str | Path, default: Any = None) -> Any:
    """Read JSON if it exists, otherwise return default."""
    src = Path(path)
    if not src.exists():
        return default
    return json.loads(src.read_text(encoding="utf-8"))


def append_csv_row(path: str | Path, row: dict[str, Any]) -> Path:
    """Append one row to a CSV, writing a header on first use."""
    return append_csv_rows(path, [row])


def append_csv_rows(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Append multiple rows to a CSV."""
    if not rows:
        return Path(path)
    out = ensure_parent(path)
    if not out.exists() or out.stat().st_size == 0:
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return out

    with out.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        existing_rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerows(rows)
    return out
