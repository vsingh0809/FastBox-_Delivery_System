# FastBox Delivery Simulator 🚚

A Python logistics simulator for the **FastBox** fictional delivery company.
Simulates one day of operations — assigning packages to agents, computing
distances, and generating a performance report.

---

## Features

| Feature | Description |
|---|---|
| JSON parsing | Robust loader with clear validation errors |
| Distance calculation | Euclidean distance with chained agent positions |
| Agent assignment | Nearest-agent-to-warehouse greedy strategy |
| Report generation | JSON report + best-agent identification |
| Random delays | Configurable seed for reproducible delay simulation |
| ASCII map | Visual route overview in the terminal |
| CSV export | Top-performer delivery breakdown |
| Full test suite | Unit + integration tests across all 10 test cases |

---

## Project Structure

```
fastbox/
├── fastbox/            # Core library package
│   ├── __init__.py     # Public API surface
│   ├── models.py       # Domain models (Warehouse, Agent, Package, …)
│   ├── loader.py       # JSON parsing & validation
│   ├── engine.py       # Distance, assignment, simulation logic
│   ├── reporter.py     # Report building & persistence
│   └── bonus.py        # Bonus features (delays, ASCII, CSV)
├── tests/
│   └── test_fastbox.py # Unit & integration tests
├── data/
│   ├── data.json       # Base case from the assignment
│   └── test_case_*.json
├── main.py             # CLI entry-point
├── pyproject.toml      # Project metadata & tooling config
└── README.md
```

---

## Quickstart

### 1. Clone & install (editable mode)

```bash
git clone https://github.com/<you>/fastbox.git
cd fastbox
pip install -e ".[dev]"
```

### 2. Run the simulation

```bash
# Base case (data/data.json → report.json)
python main.py

# Custom input / output
python main.py --input data/test_case_1.json --output results/report_1.json

# Enable all bonus features
python main.py --bonus

# ASCII map only
python main.py --ascii

# Reproducible random delays
python main.py --bonus --delay-seed 42
```

### 3. Run tests

```bash
pytest                        # all tests
pytest -v                     # verbose
pytest --cov=fastbox          # with coverage
```

---

## Example Output

```json
{
    "A1": {
        "packages_delivered": 2,
        "total_distance": 85.32,
        "efficiency": 42.66
    },
    "A2": {
        "packages_delivered": 2,
        "total_distance": 120.12,
        "efficiency": 60.06
    },
    "A3": {
        "packages_delivered": 1,
        "total_distance": 50.0,
        "efficiency": 50.0
    },
    "best_agent": "A1"
}
```

**Efficiency** = `total_distance / packages_delivered` (lower is better).
**Best agent** = the agent with the lowest efficiency score.

---

## Algorithm

1. **Parse** the input JSON and validate all fields.
2. **Assign** each package to the agent whose starting position is
   closest (Euclidean) to the package's warehouse.
3. **Simulate**: for each agent, chain deliveries in assignment order —
   the agent travels `current_position → warehouse → destination` and
   ends at the destination after each delivery.
4. **Report**: aggregate per-agent stats, identify the best agent, save
   `report.json`.

---

## Input Schema

```json
{
  "warehouses": { "W1": [x, y], ... },
  "agents":     { "A1": [x, y], ... },
  "packages": [
    { "id": "P1", "warehouse": "W1", "destination": [x, y] },
    ...
  ]
}
```

---

## CLI Reference

```
usage: fastbox [-h] [--input PATH] [--output PATH] [--bonus] [--ascii]
               [--delay-seed INT] [--csv-output PATH]

options:
  --input PATH       Input JSON file (default: data/data.json)
  --output PATH      Report output path (default: report.json)
  --bonus            Enable all bonus features
  --ascii            Print ASCII map and exit
  --delay-seed INT   Seed for reproducible random delays
  --csv-output PATH  Top-performer CSV path (default: top_performer.csv)
```

---

## License

MIT
