from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple


Coordinate = Tuple[float, float]


@dataclass(frozen=True)
class Warehouse:
    """Represents a warehouse at a fixed location."""

    id: str
    location: Coordinate

    def __repr__(self) -> str:
        return f"Warehouse({self.id}, location={self.location})"


@dataclass(frozen=True)
class Package:
    """A package waiting at a warehouse to be delivered to a destination."""

    id: str
    warehouse_id: str
    destination: Coordinate

    def __repr__(self) -> str:
        return f"Package({self.id}, warehouse={self.warehouse_id}, dest={self.destination})"


@dataclass(frozen=True)
class Agent:
    """A delivery agent starting at a given position."""

    id: str
    start_location: Coordinate

    def __repr__(self) -> str:
        return f"Agent({self.id}, start={self.start_location})"


@dataclass
class DeliveryRecord:
    """Records the outcome of a single package delivery by an agent."""

    package_id: str
    warehouse_id: str
    destination: Coordinate
    distance_to_warehouse: float
    distance_to_destination: float

    @property
    def total_distance(self) -> float:
        return self.distance_to_warehouse + self.distance_to_destination


@dataclass
class AgentReport:
    """Aggregated performance report for one agent after a day of deliveries."""

    agent_id: str
    deliveries: list[DeliveryRecord] = field(default_factory=list)

    @property
    def packages_delivered(self) -> int:
        return len(self.deliveries)

    @property
    def total_distance(self) -> float:
        return sum(d.total_distance for d in self.deliveries)

    @property
    def efficiency(self) -> float:
        """Average distance per delivery. Lower is better."""
        if self.packages_delivered == 0:
            return 0.0
        return round(self.total_distance / self.packages_delivered, 2)

    def to_dict(self) -> dict:
        return {
            "packages_delivered": self.packages_delivered,
            "total_distance": round(self.total_distance, 2),
            "efficiency": self.efficiency,
        }
