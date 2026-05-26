"""
fastbox/engine.py
-----------------
Core simulation logic: distance calculation, agent–package assignment,
and delivery simulation.

Design choices
~~~~~~~~~~~~~~
* ``euclidean_distance`` is a pure function — easy to test and swap.
* ``assign_packages`` uses a greedy nearest-agent strategy: for each
  package the closest agent (by agent-start → warehouse distance) is
  chosen.  This keeps assignment O(P × A) which is perfectly fine for
  realistic fleet sizes.
* ``simulate_deliveries`` computes per-delivery distances and returns
  structured ``DeliveryRecord`` objects consumed by the reporter.
"""

from __future__ import annotations

import math
from collections import defaultdict

from fastbox.models import (
    Agent,
    AgentReport,
    Coordinate,
    DeliveryRecord,
    Package,
    Warehouse,
)


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------


def euclidean_distance(point_a: Coordinate, point_b: Coordinate) -> float:
    """Return the Euclidean (straight-line) distance between two 2-D points.

    Parameters
    ----------
    point_a, point_b:
        Each is a ``(x, y)`` tuple of floats.
    """
    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    return math.sqrt(dx * dx + dy * dy)


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------


def assign_packages(
    packages: list[Package],
    agents: dict[str, Agent],
    warehouses: dict[str, Warehouse],
) -> dict[str, list[Package]]:
    """Assign each package to the nearest agent.

    "Nearest" is measured as the Euclidean distance from the **agent's
    starting position** to the **package's warehouse**.  Every package is
    assigned to exactly one agent; an agent may receive zero or many packages.

    Parameters
    ----------
    packages:
        All packages to be delivered today.
    agents:
        All available agents, keyed by agent-id.
    warehouses:
        All warehouses, keyed by warehouse-id.

    Returns
    -------
    Mapping of agent-id → list of packages assigned to that agent.
    """
    assignment: dict[str, list[Package]] = defaultdict(list)

    for package in packages:
        warehouse_location = warehouses[package.warehouse_id].location
        nearest_agent_id = _find_nearest_agent(warehouse_location, agents)
        assignment[nearest_agent_id].append(package)

    # Ensure every agent appears in the result (even with an empty list).
    for agent_id in agents:
        if agent_id not in assignment:
            assignment[agent_id] = []

    return dict(assignment)


def _find_nearest_agent(
    warehouse_location: Coordinate,
    agents: dict[str, Agent],
) -> str:
    """Return the id of the agent whose start position is closest to *warehouse_location*."""
    return min(
        agents.values(),
        key=lambda agent: euclidean_distance(agent.start_location, warehouse_location),
    ).id


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def simulate_deliveries(
    assignment: dict[str, list[Package]],
    agents: dict[str, Agent],
    warehouses: dict[str, Warehouse],
) -> dict[str, AgentReport]:
    """Simulate all deliveries and produce per-agent reports.

    For each agent and each assigned package the agent:
    1. Travels from its **current position** to the **warehouse**.
    2. Picks up the package and travels to the **destination**.

    The agent's current position is updated after each delivery so that
    multi-package routes are chained correctly.

    Parameters
    ----------
    assignment:
        Output from :func:`assign_packages`.
    agents:
        Agent definitions (start positions).
    warehouses:
        Warehouse definitions (locations).

    Returns
    -------
    Mapping of agent-id → :class:`AgentReport`.
    """
    reports: dict[str, AgentReport] = {}

    for agent_id, packages in assignment.items():
        agent = agents[agent_id]
        report = AgentReport(agent_id=agent_id)
        current_position: Coordinate = agent.start_location

        for package in packages:
            warehouse_location = warehouses[package.warehouse_id].location

            # Leg 1: current position → warehouse (pickup)
            dist_to_warehouse = euclidean_distance(current_position, warehouse_location)

            # Leg 2: warehouse → destination (delivery)
            dist_to_destination = euclidean_distance(warehouse_location, package.destination)

            report.deliveries.append(
                DeliveryRecord(
                    package_id=package.id,
                    warehouse_id=package.warehouse_id,
                    destination=package.destination,
                    distance_to_warehouse=dist_to_warehouse,
                    distance_to_destination=dist_to_destination,
                )
            )

            # Agent ends up at the delivery destination, ready for the next job.
            current_position = package.destination

        reports[agent_id] = report

    return reports
