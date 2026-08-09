# Repository Guidelines

## Project Structure & Module Organization
- `control/`: MPC controllers/optimizers (`psdf_optimizer*.py`, `nmpc_optimizer*.py`, `dcbf_optimizer*.py`).
- `models/`: Dynamics + geometry (`dd.py`, `geometry_utils.py`, `psdf_wrapper.py`).
- `planning/`, `sim/`: Planning utilities and simulation loop.
- `config.yaml`: Scenario configs; outputs go to `animations/`, `figures/`, `data/`.
- Tests: top-level `test_*.py` scripts.

## Build, Test, and Development Commands
- Python 3.10+; minimal deps: `pip install numpy matplotlib pandas pyyaml torch` (+ optional `casadi`, `acados`).
- NMPC: `python test_nmpc.py -c config.yaml` or quick `python test_nmpc.py --single`.
- Other runs: `python test_hybrid_astar.py`, `python test_psdf.py`.

## Coding Style & Naming Conventions
- PEP 8, 4-space indents, type hints where practical.
- `snake_case` files/functions, `CamelCase` classes, `UPPER_SNAKE_CASE` constants.

