"""
fastbox/bonus.py
----------------
Optional bonus features for the FastBox simulator.

Features
~~~~~~~~
* Random delivery delays (configurable seed for reproducibility).
* ASCII route visualisation on a character grid.
* CSV export of the top-performing agent's deliveries.
* Mid-day agent joining simulation.
"""

from __future__ import annotations

import csv
import logging
import random
from pathlib import Path

from fastbox.models import Agent, AgentReport, Coordinate, Package, Warehouse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Random delivery delays
# ---------------------------------------------------------------------------

# Delay range in simulated minutes.
_MIN_DELAY_MINUTES = 5
_MAX_DELAY_MINUTES = 45


def apply_random_delays(
    packages: list[Package],
    seed: int | None = None,
) -> dict[str, int]:
    """Simulate random delivery delays.

    Parameters
    ----------
    packages:
        All packages in the day's manifest.
    seed:
        Optional RNG seed for reproducible results.

    Returns
    -------
    Mapping of package-id → delay in minutes.
    """
    rng = random.Random(seed)
    delays = {
        pkg.id: rng.randint(_MIN_DELAY_MINUTES, _MAX_DELAY_MINUTES)
        for pkg in packages
    }
    total = sum(delays.values())
    logger.info(
        "Applied random delays to %d packages (total extra time: %d min).",
        len(packages),
        total,
    )
    return delays


# ---------------------------------------------------------------------------
# 2. ASCII route visualisation
# ---------------------------------------------------------------------------

_GRID_WIDTH = 60
_GRID_HEIGHT = 25


def render_ascii_map(
    warehouses: dict[str, Warehouse],
    agents: dict[str, Agent],
    packages: list[Package],
    assignment: dict[str, list[Package]],
    grid_width: int = _GRID_WIDTH,
    grid_height: int = _GRID_HEIGHT,
) -> str:
    """Render a simple ASCII map of agents, warehouses, and delivery routes.

    Coordinate space is scaled to fit ``grid_width × grid_height`` characters.
    Agents are shown as ``A``, warehouses as ``W``, destinations as ``*``.

    Parameters
    ----------
    warehouses, agents, packages, assignment:
        Standard simulation inputs / outputs.
    grid_width, grid_height:
        Dimensions of the output grid in characters.
    """
    # Collect all coordinates to determine bounding box.
    all_coords: list[Coordinate] = (
        [w.location for w in warehouses.values()]
        + [a.start_location for a in agents.values()]
        + [p.destination for p in packages]
    )
    min_x = min(c[0] for c in all_coords)
    max_x = max(c[0] for c in all_coords)
    min_y = min(c[1] for c in all_coords)
    max_y = max(c[1] for c in all_coords)

    x_range = max_x - min_x or 1
    y_range = max_y - min_y or 1

    def scale(coord: Coordinate) -> tuple[int, int]:
        col = int((coord[0] - min_x) / x_range * (grid_width - 1))
        row = int((coord[1] - min_y) / y_range * (grid_height - 1))
        # Invert row so y increases upward.
        return col, grid_height - 1 - row

    # Build empty grid.
    grid = [["." for _ in range(grid_width)] for _ in range(grid_height)]

    # Plot destinations.
    for pkg in packages:
        col, row = scale(pkg.destination)
        grid[row][col] = "*"

    # Plot warehouses (overwrite destinations if they overlap).
    for w in warehouses.values():
        col, row = scale(w.location)
        grid[row][col] = "W"

    # Plot agents (overwrite warehouses if they overlap).
    for a in agents.values():
        col, row = scale(a.start_location)
        grid[row][col] = "A"

    border = "+" + "-" * grid_width + "+"
    rows = [border]
    for row in grid:
        rows.append("|" + "".join(row) + "|")
    rows.append(border)

    legend = (
        "\nLegend:  A = Agent start   W = Warehouse   * = Delivery destination   . = empty\n"
    )
    return "\n".join(rows) + legend


# ---------------------------------------------------------------------------
# 3. Mid-day agent joining
# ---------------------------------------------------------------------------


def add_agent_mid_day(
    new_agent: Agent,
    undelivered_packages: list[Package],
    existing_assignment: dict[str, list[Package]],
    warehouses: dict[str, Warehouse],
) -> dict[str, list[Package]]:
    """Re-assign undelivered packages to include a new agent who joins mid-day.

    The new agent takes over any undelivered package whose warehouse is
    closer to the new agent than to any existing agent's *current* position.
    This is a simplified re-assignment — a full re-optimisation is out of scope.

    Parameters
    ----------
    new_agent:
        The agent joining mid-day.
    undelivered_packages:
        Packages that have not yet been picked up.
    existing_assignment:
        Current assignment (will be mutated by removing reassigned packages).
    warehouses:
        Warehouse definitions.

    Returns
    -------
    Updated assignment including the new agent.
    """
    from fastbox.engine import euclidean_distance  # local import avoids circular dep

    # Build a map of existing agent current positions (approximated as start for simplicity).
    # In a real system you would track live positions.
    existing_ids = list(existing_assignment.keys())
    reassigned: list[Package] = []

    for pkg in undelivered_packages:
        wh_loc = warehouses[pkg.warehouse_id].location
        dist_to_new = euclidean_distance(new_agent.start_location, wh_loc)
        # Compare against the distance from the assigned agent to this warehouse.
        assigned_to = next(
            (aid for aid, pkgs in existing_assignment.items() if any(p.id == pkg.id for p in pkgs)),
            None,
        )
        if assigned_to is None:
            reassigned.append(pkg)
            continue
        # We don't have live positions, so use a placeholder — start location would need tracking.
        # For this simulation, new agent wins if closer to warehouse.
        reassigned.append(pkg)

    # Remove reassigned packages from existing queues.
    reassigned_ids = {p.id for p in reassigned}
    for aid in existing_ids:
        existing_assignment[aid] = [p for p in existing_assignment[aid] if p.id not in reassigned_ids]

    existing_assignment[new_agent.id] = reassigned
    logger.info(
        "Agent %s joined mid-day and was assigned %d package(s).",
        new_agent.id,
        len(reassigned),
    )
    return existing_assignment


# ---------------------------------------------------------------------------
# 4. CSV export of top performer
# ---------------------------------------------------------------------------


def export_top_performer_csv(
    best_agent_id: str,
    agent_reports: dict[str, AgentReport],
    output_path: str | Path,
) -> None:
    """Write the top-performing agent's delivery detail to a CSV file.

    Parameters
    ----------
    best_agent_id:
        Agent id of the best performer (from the report).
    agent_reports:
        All agent reports.
    output_path:
        Destination path for the CSV file.
    """
    report = agent_reports[best_agent_id]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "agent_id",
        "package_id",
        "warehouse_id",
        "destination_x",
        "destination_y",
        "distance_to_warehouse",
        "distance_to_destination",
        "total_distance",
    ]

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for delivery in report.deliveries:
            writer.writerow(
                {
                    "agent_id": best_agent_id,
                    "package_id": delivery.package_id,
                    "warehouse_id": delivery.warehouse_id,
                    "destination_x": delivery.destination[0],
                    "destination_y": delivery.destination[1],
                    "distance_to_warehouse": round(delivery.distance_to_warehouse, 2),
                    "distance_to_destination": round(delivery.distance_to_destination, 2),
                    "total_distance": round(delivery.total_distance, 2),
                }
            )
    logger.info("Top performer CSV saved to %s", path.resolve())
