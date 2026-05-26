"""
main.py
-------
CLI entry-point for the FastBox delivery simulator.

Usage
-----
    python main.py                          # uses data/data.json, saves report.json
    python main.py --input data/data.json   # explicit input
    python main.py --input data/data.json --output results/report.json
    python main.py --bonus                  # enable all bonus features
    python main.py --ascii                  # show ASCII map only
    python main.py --delay-seed 42          # reproducible random delays
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from fastbox import (
    assign_packages,
    build_report,
    load_data,
    print_report,
    save_report,
    simulate_deliveries,
)
from fastbox.bonus import (
    apply_random_delays,
    export_top_performer_csv,
    render_ascii_map,
)

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fastbox",
        description="FastBox Delivery Simulator — simulate one day of logistics operations.",
    )
    parser.add_argument(
        "--input",
        default="data/data.json",
        metavar="PATH",
        help="Path to the input JSON file (default: data/data.json).",
    )
    parser.add_argument(
        "--output",
        default="report.json",
        metavar="PATH",
        help="Path where the report JSON will be saved (default: report.json).",
    )
    parser.add_argument(
        "--bonus",
        action="store_true",
        help="Enable all bonus features (delays, ASCII map, CSV export).",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="Print an ASCII map of routes and exit.",
    )
    parser.add_argument(
        "--delay-seed",
        type=int,
        default=None,
        metavar="INT",
        help="Seed for random delay generation (omit for non-deterministic delays).",
    )
    parser.add_argument(
        "--csv-output",
        default="top_performer.csv",
        metavar="PATH",
        help="Path for top-performer CSV export (default: top_performer.csv).",
    )
    return parser


# ---------------------------------------------------------------------------
# Main simulation pipeline
# ---------------------------------------------------------------------------


def run(
    input_path: str,
    output_path: str,
    bonus: bool = False,
    ascii_only: bool = False,
    delay_seed: int | None = None,
    csv_output: str = "top_performer.csv",
) -> dict:
    """Execute the full simulation pipeline and return the report dict.

    Parameters
    ----------
    input_path:
        Path to the input JSON file.
    output_path:
        Destination for the report JSON.
    bonus:
        Enable bonus features (delays, ASCII map, CSV export).
    ascii_only:
        Print ASCII map and return without saving a report.
    delay_seed:
        RNG seed for reproducible delay simulation.
    csv_output:
        Path for the top-performer CSV.

    Returns
    -------
    The compiled report dictionary.
    """
    # --- Step 1: Load and parse input data -----------------------------------
    logger.info("Loading data from '%s' …", input_path)
    warehouses, agents, packages = load_data(input_path)
    logger.info(
        "Loaded %d warehouse(s), %d agent(s), %d package(s).",
        len(warehouses),
        len(agents),
        len(packages),
    )

    # --- Bonus: random delays ------------------------------------------------
    if bonus or delay_seed is not None:
        delays = apply_random_delays(packages, seed=delay_seed)
        logger.info("Delivery delays (minutes): %s", delays)

    # --- Step 2: Assign packages to nearest agents ---------------------------
    logger.info("Assigning packages to nearest agents …")
    assignment = assign_packages(packages, agents, warehouses)
    for agent_id, pkgs in sorted(assignment.items()):
        pkg_ids = [p.id for p in pkgs]
        logger.info("  %s → %s", agent_id, pkg_ids if pkg_ids else "(no packages)")

    # --- Bonus: ASCII map ----------------------------------------------------
    if bonus or ascii_only:
        print("\n" + render_ascii_map(warehouses, agents, packages, assignment))
        if ascii_only:
            return {}

    # --- Step 3: Simulate deliveries -----------------------------------------
    logger.info("Simulating deliveries …")
    agent_reports = simulate_deliveries(assignment, agents, warehouses)

    # --- Step 4: Build and save report ---------------------------------------
    report = build_report(agent_reports)
    print_report(report)
    save_report(report, output_path)

    # --- Bonus: CSV export of best agent -------------------------------------
    if bonus and report.get("best_agent"):
        export_top_performer_csv(report["best_agent"], agent_reports, csv_output)

    # --- Validation: all packages accounted for ------------------------------
    total_delivered = sum(r["packages_delivered"] for key, r in report.items() if key != "best_agent")
    if total_delivered != len(packages):
        logger.error(
            "Delivery count mismatch: delivered %d but expected %d!",
            total_delivered,
            len(packages),
        )
    else:
        logger.info("✓ All %d package(s) delivered successfully.", total_delivered)

    return report


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        run(
            input_path=args.input,
            output_path=args.output,
            bonus=args.bonus,
            ascii_only=args.ascii,
            delay_seed=args.delay_seed,
            csv_output=args.csv_output,
        )
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        sys.exit(1)
    except ValueError as exc:
        logger.error("Invalid input data: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
