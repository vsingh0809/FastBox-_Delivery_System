from __future__ import annotations
import json
import logging
from pathlib import Path
from fastbox.models import AgentReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_report(agent_reports: dict[str, AgentReport]) -> dict:
    """Compile agent reports into the final output dictionary.

    Parameters
    ----------
    agent_reports:
        Mapping of agent-id → :class:`AgentReport`.

    Returns
    -------
    A plain dictionary suitable for JSON serialisation.
    """
    report: dict = {}

    for agent_id in sorted(agent_reports):
        report[agent_id] = agent_reports[agent_id].to_dict()

    report["best_agent"] = _find_best_agent(agent_reports)
    return report


def save_report(report: dict, output_path: str | Path) -> None:
    """Write *report* as pretty-printed JSON to *output_path*.

    Creates parent directories if they do not already exist.

    Parameters
    ----------
    report:
        The compiled report dictionary from :func:`build_report`.
    output_path:
        Destination file path (will be created or overwritten).
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=4)
    logger.info("Report saved to %s", path.resolve())


def print_report(report: dict) -> None:
    """Pretty-print the report to stdout."""
    print(json.dumps(report, indent=4))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_best_agent(agent_reports: dict[str, AgentReport]) -> str | None:
    """Return the agent-id with the lowest efficiency score (best performer).

    Agents with zero deliveries are excluded — they have not done any work.
    Returns ``None`` if no agent delivered anything.
    """
    active_agents = {aid: r for aid, r in agent_reports.items() if r.packages_delivered > 0}
    if not active_agents:
        logger.warning("No packages were delivered — cannot determine best agent.")
        return None
    return min(active_agents, key=lambda aid: active_agents[aid].efficiency)
