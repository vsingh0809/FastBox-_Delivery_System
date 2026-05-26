from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from fastbox.models import Agent, Package, Warehouse


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_data(path: str | Path) -> tuple[dict[str, Warehouse], dict[str, Agent], list[Package]]:
    """Parse a FastBox JSON file and return domain objects.

    Parameters
    ----------
    path:
        Path to the JSON input file.

    Returns
    -------
    warehouses:
        Mapping of warehouse-id → Warehouse.
    agents:
        Mapping of agent-id → Agent.
    packages:
        Ordered list of Package objects.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the JSON structure is invalid or required fields are missing.
    """
    raw = _read_json(path)
    _validate_top_level(raw)

    warehouses = _parse_warehouses(raw["warehouses"])
    agents = _parse_agents(raw["agents"])
    packages = _parse_packages(raw["packages"], warehouses)

    return warehouses, agents, packages


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_json(path: str | Path) -> Any:
    """Read and deserialise a JSON file."""
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")
    with filepath.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _validate_top_level(raw: Any) -> None:
    """Ensure the top-level keys are present and are the right types."""
    if not isinstance(raw, dict):
        raise ValueError("Root JSON element must be an object.")
    for key in ("warehouses", "agents", "packages"):
        if key not in raw:
            raise ValueError(f"Missing required top-level key: '{key}'.")


def _parse_coordinate(value: Any, context: str) -> tuple[float, float]:
    """Convert a raw JSON value to a (x, y) float tuple.

    Parameters
    ----------
    value:
        The raw value from JSON (expected to be a list of two numbers).
    context:
        Human-readable label used in error messages.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{context}: coordinate must be a list of exactly two numbers, got {value!r}.")
    x, y = value
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        raise ValueError(f"{context}: coordinate values must be numbers, got [{x!r}, {y!r}].")
    return float(x), float(y)


def _parse_warehouses(raw: Any) -> dict[str, Warehouse]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError("'warehouses' must be a non-empty object.")
    return {
        wid: Warehouse(id=wid, location=_parse_coordinate(loc, f"Warehouse {wid}"))
        for wid, loc in raw.items()
    }


def _parse_agents(raw: Any) -> dict[str, Agent]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError("'agents' must be a non-empty object.")
    return {
        aid: Agent(id=aid, start_location=_parse_coordinate(loc, f"Agent {aid}"))
        for aid, loc in raw.items()
    }


def _parse_packages(raw: Any, warehouses: dict[str, Warehouse]) -> list[Package]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("'packages' must be a non-empty array.")
    packages = []
    for i, item in enumerate(raw):
        context = f"Package at index {i}"
        if not isinstance(item, dict):
            raise ValueError(f"{context}: must be an object.")
        for field in ("id", "warehouse", "destination"):
            if field not in item:
                raise ValueError(f"{context}: missing required field '{field}'.")
        pid = item["id"]
        wid = item["warehouse"]
        if wid not in warehouses:
            raise ValueError(f"Package '{pid}' references unknown warehouse '{wid}'.")
        dest = _parse_coordinate(item["destination"], f"Package {pid} destination")
        packages.append(Package(id=pid, warehouse_id=wid, destination=dest))
    return packages
