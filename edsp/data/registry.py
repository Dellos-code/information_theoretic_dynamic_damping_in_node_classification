"""Access to ``data/REGISTRY.json`` (Master Prompt §3.1/§3.3).

JSON rather than YAML so that the registry needs no extra dependency; see
docs/SPEC_AMBIGUITIES.md.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "data" / "REGISTRY.json"


@lru_cache(maxsize=1)
def registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def dataset_entry(name: str) -> dict:
    try:
        return registry()["datasets"][name]
    except KeyError as exc:
        raise KeyError(f"{name!r} is not in data/REGISTRY.json") from exc


def candidate_urls(name: str) -> list[str]:
    return list(dataset_entry(name)["routes"])


def cell_of(name: str) -> str:
    return str(dataset_entry(name)["cell"])


def datasets_with_loader() -> list[str]:
    return sorted(k for k, v in registry()["datasets"].items() if v.get("loader"))
