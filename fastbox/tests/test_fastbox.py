"""
tests/test_fastbox.py
---------------------
Unit and integration tests for the FastBox delivery simulator.

Run with:
    pytest tests/ -v
    pytest tests/ -v --tb=short   # shorter tracebacks
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from fastbox.engine import assign_packages, euclidean_distance, simulate_deliveries
from fastbox.loader import load_data
from fastbox.models import Agent, AgentReport, DeliveryRecord, Package, Warehouse
from fastbox.reporter import build_report, save_report


# ---------------------------------------------------------------------------
# Fixtures — shared test data
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_warehouses() -> dict[str, Warehouse]:
    return {
        "W1": Warehouse(id="W1", location=(0.0, 0.0)),
        "W2": Warehouse(id="W2", location=(50.0, 75.0)),
        "W3": Warehouse(id="W3", location=(100.0, 25.0)),
    }


@pytest.fixture
def sample_agents() -> dict[str, Agent]:
    return {
        "A1": Agent(id="A1", start_location=(5.0, 5.0)),
        "A2": Agent(id="A2", start_location=(60.0, 60.0)),
        "A3": Agent(id="A3", start_location=(95.0, 30.0)),
    }


@pytest.fixture
def sample_packages() -> list[Package]:
    return [
        Package(id="P1", warehouse_id="W1", destination=(30.0, 40.0)),
        Package(id="P2", warehouse_id="W2", destination=(70.0, 90.0)),
        Package(id="P3", warehouse_id="W3", destination=(105.0, 20.0)),
        Package(id="P4", warehouse_id="W1", destination=(10.0, 10.0)),
        Package(id="P5", warehouse_id="W2", destination=(40.0, 80.0)),
    ]


@pytest.fixture
def base_case_json(tmp_path: Path) -> Path:
    """Write the assignment base-case JSON to a temp file."""
    data = {
        "warehouses": {"W1": [0, 0], "W2": [50, 75], "W3": [100, 25]},
        "agents": {"A1": [5, 5], "A2": [60, 60], "A3": [95, 30]},
        "packages": [
            {"id": "P1", "warehouse": "W1", "destination": [30, 40]},
            {"id": "P2", "warehouse": "W2", "destination": [70, 90]},
            {"id": "P3", "warehouse": "W3", "destination": [105, 20]},
            {"id": "P4", "warehouse": "W1", "destination": [10, 10]},
            {"id": "P5", "warehouse": "W2", "destination": [40, 80]},
        ],
    }
    p = tmp_path / "base_case.json"
    p.write_text(json.dumps(data))
    return p


# ---------------------------------------------------------------------------
# Tests: euclidean_distance
# ---------------------------------------------------------------------------


class TestEuclideanDistance:
    def test_same_point_is_zero(self):
        assert euclidean_distance((3.0, 4.0), (3.0, 4.0)) == 0.0

    def test_origin_to_345_triangle(self):
        # 3-4-5 right triangle
        assert math.isclose(euclidean_distance((0.0, 0.0), (3.0, 4.0)), 5.0)

    def test_symmetry(self):
        a, b = (10.0, 20.0), (30.0, 50.0)
        assert euclidean_distance(a, b) == euclidean_distance(b, a)

    def test_horizontal_line(self):
        assert math.isclose(euclidean_distance((0.0, 0.0), (10.0, 0.0)), 10.0)

    def test_vertical_line(self):
        assert math.isclose(euclidean_distance((0.0, 0.0), (0.0, 7.0)), 7.0)

    def test_float_coordinates(self):
        # sqrt((1.5)^2 + (2.0)^2) = sqrt(2.25 + 4) = sqrt(6.25) = 2.5
        assert math.isclose(euclidean_distance((0.0, 0.0), (1.5, 2.0)), 2.5)


# ---------------------------------------------------------------------------
# Tests: assign_packages
# ---------------------------------------------------------------------------


class TestAssignPackages:
    def test_every_package_assigned(self, sample_warehouses, sample_agents, sample_packages):
        assignment = assign_packages(sample_packages, sample_agents, sample_warehouses)
        assigned_packages = [p for pkgs in assignment.values() for p in pkgs]
        assert len(assigned_packages) == len(sample_packages)

    def test_every_agent_present_in_result(self, sample_warehouses, sample_agents, sample_packages):
        assignment = assign_packages(sample_packages, sample_agents, sample_warehouses)
        assert set(assignment.keys()) == set(sample_agents.keys())

    def test_no_duplicate_assignments(self, sample_warehouses, sample_agents, sample_packages):
        assignment = assign_packages(sample_packages, sample_agents, sample_warehouses)
        all_ids = [p.id for pkgs in assignment.values() for p in pkgs]
        assert len(all_ids) == len(set(all_ids))

    def test_nearest_agent_gets_package(self):
        """W1 is at (0,0).  A1 is at (1,0).  A2 is at (100,100).  A1 should win."""
        warehouses = {"W1": Warehouse("W1", (0.0, 0.0))}
        agents = {
            "A1": Agent("A1", (1.0, 0.0)),
            "A2": Agent("A2", (100.0, 100.0)),
        }
        packages = [Package("P1", "W1", (5.0, 5.0))]
        assignment = assign_packages(packages, agents, warehouses)
        assert "P1" in [p.id for p in assignment["A1"]]

    def test_single_agent_gets_all_packages(self):
        warehouses = {"W1": Warehouse("W1", (0.0, 0.0))}
        agents = {"A1": Agent("A1", (0.0, 0.0))}
        packages = [
            Package("P1", "W1", (1.0, 0.0)),
            Package("P2", "W1", (2.0, 0.0)),
        ]
        assignment = assign_packages(packages, agents, warehouses)
        assert len(assignment["A1"]) == 2


# ---------------------------------------------------------------------------
# Tests: simulate_deliveries
# ---------------------------------------------------------------------------


class TestSimulateDeliveries:
    def test_all_packages_in_reports(self, sample_warehouses, sample_agents, sample_packages):
        assignment = assign_packages(sample_packages, sample_agents, sample_warehouses)
        reports = simulate_deliveries(assignment, sample_agents, sample_warehouses)
        total = sum(r.packages_delivered for r in reports.values())
        assert total == len(sample_packages)

    def test_total_distance_is_positive(self, sample_warehouses, sample_agents, sample_packages):
        assignment = assign_packages(sample_packages, sample_agents, sample_warehouses)
        reports = simulate_deliveries(assignment, sample_agents, sample_warehouses)
        for report in reports.values():
            if report.packages_delivered > 0:
                assert report.total_distance > 0

    def test_single_delivery_distance(self):
        """
        Agent at (0,0), warehouse at (3,4), destination at (6,8).
        dist_to_warehouse = 5.0 (3-4-5 triangle).
        dist_to_destination = sqrt((6-3)^2 + (8-4)^2) = 5.0.
        total = 10.0.
        """
        warehouses = {"W1": Warehouse("W1", (3.0, 4.0))}
        agents = {"A1": Agent("A1", (0.0, 0.0))}
        packages = [Package("P1", "W1", (6.0, 8.0))]
        assignment = {"A1": packages}
        reports = simulate_deliveries(assignment, agents, warehouses)
        assert math.isclose(reports["A1"].total_distance, 10.0, rel_tol=1e-9)

    def test_efficiency_calculation(self):
        warehouses = {"W1": Warehouse("W1", (3.0, 4.0))}
        agents = {"A1": Agent("A1", (0.0, 0.0))}
        packages = [Package("P1", "W1", (6.0, 8.0))]
        assignment = {"A1": packages}
        reports = simulate_deliveries(assignment, agents, warehouses)
        # 1 delivery, total 10.0 → efficiency 10.0
        assert math.isclose(reports["A1"].efficiency, 10.0, rel_tol=1e-3)

    def test_chained_positions(self):
        """After delivering P1, the agent should start at P1's destination for P2."""
        warehouses = {
            "W1": Warehouse("W1", (0.0, 0.0)),
            "W2": Warehouse("W2", (10.0, 0.0)),
        }
        agents = {"A1": Agent("A1", (0.0, 0.0))}
        packages = [
            Package("P1", "W1", (5.0, 0.0)),   # drop at (5,0)
            Package("P2", "W2", (15.0, 0.0)),  # pick up from (10,0), drop at (15,0)
        ]
        assignment = {"A1": packages}
        reports = simulate_deliveries(assignment, agents, warehouses)
        deliveries = reports["A1"].deliveries
        # P2: agent is at (5,0) after P1, warehouse at (10,0) → dist = 5
        assert math.isclose(deliveries[1].distance_to_warehouse, 5.0, rel_tol=1e-9)

    def test_agent_with_no_packages_has_zero_distance(self):
        warehouses = {"W1": Warehouse("W1", (0.0, 0.0))}
        agents = {"A1": Agent("A1", (0.0, 0.0))}
        assignment = {"A1": []}
        reports = simulate_deliveries(assignment, agents, warehouses)
        assert reports["A1"].total_distance == 0.0
        assert reports["A1"].packages_delivered == 0


# ---------------------------------------------------------------------------
# Tests: AgentReport
# ---------------------------------------------------------------------------


class TestAgentReport:
    def _make_report(self, n: int, dist_per_delivery: float) -> AgentReport:
        report = AgentReport(agent_id="A1")
        for i in range(n):
            report.deliveries.append(
                DeliveryRecord(
                    package_id=f"P{i}",
                    warehouse_id="W1",
                    destination=(0.0, 0.0),
                    distance_to_warehouse=dist_per_delivery / 2,
                    distance_to_destination=dist_per_delivery / 2,
                )
            )
        return report

    def test_packages_delivered_count(self):
        report = self._make_report(3, 10.0)
        assert report.packages_delivered == 3

    def test_total_distance(self):
        report = self._make_report(3, 10.0)
        assert math.isclose(report.total_distance, 30.0)

    def test_efficiency(self):
        report = self._make_report(3, 10.0)
        assert math.isclose(report.efficiency, 10.0)

    def test_zero_deliveries_efficiency(self):
        report = AgentReport(agent_id="A1")
        assert report.efficiency == 0.0

    def test_to_dict_shape(self):
        report = self._make_report(2, 20.0)
        d = report.to_dict()
        assert set(d.keys()) == {"packages_delivered", "total_distance", "efficiency"}


# ---------------------------------------------------------------------------
# Tests: build_report
# ---------------------------------------------------------------------------


class TestBuildReport:
    def test_best_agent_selected(self, sample_warehouses, sample_agents, sample_packages):
        assignment = assign_packages(sample_packages, sample_agents, sample_warehouses)
        agent_reports = simulate_deliveries(assignment, sample_agents, sample_warehouses)
        report = build_report(agent_reports)
        assert "best_agent" in report
        assert report["best_agent"] in sample_agents

    def test_all_agents_in_report(self, sample_warehouses, sample_agents, sample_packages):
        assignment = assign_packages(sample_packages, sample_agents, sample_warehouses)
        agent_reports = simulate_deliveries(assignment, sample_agents, sample_warehouses)
        report = build_report(agent_reports)
        for aid in sample_agents:
            assert aid in report

    def test_total_packages_match(self, sample_warehouses, sample_agents, sample_packages):
        assignment = assign_packages(sample_packages, sample_agents, sample_warehouses)
        agent_reports = simulate_deliveries(assignment, sample_agents, sample_warehouses)
        report = build_report(agent_reports)
        total = sum(v["packages_delivered"] for k, v in report.items() if k != "best_agent")
        assert total == len(sample_packages)


# ---------------------------------------------------------------------------
# Tests: loader
# ---------------------------------------------------------------------------


class TestLoader:
    def test_load_base_case(self, base_case_json):
        warehouses, agents, packages = load_data(base_case_json)
        assert len(warehouses) == 3
        assert len(agents) == 3
        assert len(packages) == 5

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_data(tmp_path / "nonexistent.json")

    def test_unknown_warehouse_raises(self, tmp_path):
        data = {
            "warehouses": {"W1": [0, 0]},
            "agents": {"A1": [0, 0]},
            "packages": [{"id": "P1", "warehouse": "W_UNKNOWN", "destination": [1, 1]}],
        }
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="unknown warehouse"):
            load_data(p)

    def test_missing_top_level_key_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"warehouses": {}, "agents": {}}))
        with pytest.raises(ValueError, match="packages"):
            load_data(p)

    def test_invalid_coordinate_raises(self, tmp_path):
        data = {
            "warehouses": {"W1": [0]},   # only one value — invalid
            "agents": {"A1": [0, 0]},
            "packages": [],
        }
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(data))
        with pytest.raises(ValueError):
            load_data(p)

    def test_coordinates_are_floats(self, base_case_json):
        warehouses, agents, _ = load_data(base_case_json)
        for w in warehouses.values():
            assert all(isinstance(v, float) for v in w.location)
        for a in agents.values():
            assert all(isinstance(v, float) for v in a.start_location)


# ---------------------------------------------------------------------------
# Tests: save_report
# ---------------------------------------------------------------------------


class TestSaveReport:
    def test_report_file_created(self, tmp_path, sample_warehouses, sample_agents, sample_packages):
        assignment = assign_packages(sample_packages, sample_agents, sample_warehouses)
        agent_reports = simulate_deliveries(assignment, sample_agents, sample_warehouses)
        report = build_report(agent_reports)
        out = tmp_path / "report.json"
        save_report(report, out)
        assert out.exists()

    def test_report_is_valid_json(self, tmp_path, sample_warehouses, sample_agents, sample_packages):
        assignment = assign_packages(sample_packages, sample_agents, sample_warehouses)
        agent_reports = simulate_deliveries(assignment, sample_agents, sample_warehouses)
        report = build_report(agent_reports)
        out = tmp_path / "report.json"
        save_report(report, out)
        loaded = json.loads(out.read_text())
        assert "best_agent" in loaded


# ---------------------------------------------------------------------------
# Integration tests: all 10 test case files
# ---------------------------------------------------------------------------


TEST_CASE_DIR = Path(__file__).parent.parent / "data"


@pytest.mark.parametrize(
    "json_file",
    sorted(TEST_CASE_DIR.glob("test_case_*.json")),
    ids=lambda p: p.name,
)
def test_all_test_cases(json_file: Path):
    """Full pipeline integration test for every provided test-case file."""
    warehouses, agents, packages = load_data(json_file)

    # Ensure loading succeeds and counts are sane.
    assert warehouses
    assert agents
    assert packages

    assignment = assign_packages(packages, agents, warehouses)
    assert set(assignment.keys()) == set(agents.keys())

    agent_reports = simulate_deliveries(assignment, agents, warehouses)
    report = build_report(agent_reports)

    # Every package must be accounted for.
    total = sum(v["packages_delivered"] for k, v in report.items() if k != "best_agent")
    assert total == len(packages), (
        f"{json_file.name}: delivered {total} but expected {len(packages)}"
    )

    # Best agent must be a valid agent id.
    assert report["best_agent"] in agents

    # Efficiency values must be non-negative.
    for key, val in report.items():
        if key == "best_agent":
            continue
        assert val["efficiency"] >= 0
        assert val["total_distance"] >= 0
