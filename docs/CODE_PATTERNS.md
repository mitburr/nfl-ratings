# 🏈 NFL Elo Rating System – Code Style & Architecture Guide

## 0. Notes for AI Agents

If you are an AI agent working on code for this project. 
Hello! Welcome! 
- When responding to users about code, make sure to always provide complete code files which can be run. Users would prefer to copy and paste complete files, rather than digging to find which lines need to be changed. 
- Make sure to be clear about what file you are working on. Users think in atomic file units, not complete code projects, so label where files should go in the directory structure. 

## 1. Overall Philosophy

> “Readable, Reproducible, Reusable.”

This project is **research-grade**, not production SaaS — so:

* **Clarity > cleverness.**
* **Determinism > speed** (unless proven bottleneck).
* **Reproducibility** is non-negotiable: every experiment must be reproducible from config + code commit.

---

## 2. Project Layout

```
nfl-ratings/
│
├── src/
│   ├── models/
│   │   ├── base.py              # BasePredictor, common interface
│   │   ├── elo.py               # EloPredictor implementation
│   │   └── vector_enhanced.py   # VectorEnhancedPredictor
│   │
│   ├── analysis/
│   │   ├── team_vectors.py      # Vector computation (no leakage)
│   │   └── visualizations.py
│   │
│   ├── evaluation/
│   │   ├── evaluator.py         # ModelEvaluator
│   │   └── metrics.py           # Metrics registry
│   │
│   ├── database/
│   │   ├── db_manager.py        # SQLite + PostgreSQL utilities
│   │   └── schema.sql
│   │
│   └── utils/
│       ├── config.py            # YAML loader + overrides + validation
│       ├── cache.py             # EloCache
│       └── logging.py
│
├── scripts/
│   ├── predict_game.py
│   ├── run_full_analysis.py
│   ├── test_vectors_proper.py
│   ├── analyze_vectors.py
│   └── experiments/
│       └── runner.py            # Experiment runner + overrides
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── plots/
│   └── experiments/
│
├── config/
│   ├── base.yaml
│   └── variants/
│
├── tests/
│   ├── test_elo.py
│   ├── test_vectors.py
│   └── test_evaluator.py
│
├── requirements.txt
└── README.md
```

---

## 3. Naming Conventions

| Entity Type            | Convention                   | Example                                     |
| ---------------------- | ---------------------------- | ------------------------------------------- |
| **Classes**            | `PascalCase`                 | `EloPredictor`, `EloCache`                  |
| **Functions**          | `snake_case`                 | `calculate_team_vectors`, `apply_overrides` |
| **Variables**          | `snake_case`                 | `through_week`, `boost_weight`              |
| **Constants**          | `UPPER_SNAKE_CASE`           | `DEFAULT_HOME_ADVANTAGE`                    |
| **Modules**            | `lowercase_with_underscores` | `team_vectors.py`                           |
| **Experiment Configs** | kebab-case YAML filenames    | `vector-boost-0.3.yaml`                     |

---

## 4. Imports

Always use **absolute imports** within `src/`.
✅ Good:

```python
from src.models.elo import EloPredictor
from src.utils.config import load_config
```

❌ Bad:

```python
from ..models.elo import EloPredictor
```

**Import order:**

1. Standard library
2. Third-party libraries
3. Internal modules (grouped by layer)

Example:

```python
import os
import json
from typing import List, Dict

import pandas as pd
import yaml

from src.models.elo import EloPredictor
from src.utils.cache import EloCache
```

---

## 5. Docstrings & Comments

Use **Google-style docstrings**.
Include **Args**, **Returns**, and **Example** if applicable.

```python
def calculate_team_vectors(plays: pd.DataFrame, through_week: int) -> pd.DataFrame:
    """Compute team offensive/defensive vectors through a given week.

    Args:
        plays: Play-by-play DataFrame from nflverse.
        through_week: Last week to include (inclusive).

    Returns:
        A DataFrame with per-team aggregated metrics.
    """
```

* Use comments **only for intent**, not mechanics.
* Prefer **short, meaningful names** over comments explaining trivial code.

---

## 6. Configuration Style

### 6.1 YAML Structure

Each YAML file should have:

```yaml
name: "elo_baseline_2025"
model_type: "elo"
parameters:
  k_factor: 20
  home_advantage: 40
  regress_to: 1500
metrics: [accuracy, brier_score]
data:
  seasons: [2024, 2025]
  through_week: 9
output:
  save_predictions: true
```

### 6.2 Command-Line Overrides

* Always support `--override key=value`.
* Override keys can use dot-notation: `elo.k_factor=25`.
* Values must infer type via `yaml.safe_load`.

### 6.3 Validation

* Each config must be validated by `validate_config(config)` before use.
* Raise `ValueError` with clear, human-readable messages.

---

## 7. Logging & Output

Use the built-in `logging` module:

```python
import logging
logger = logging.getLogger(__name__)

logger.info("Training Elo model for %s", season)
logger.debug("Elo update: %s vs %s, delta=%f", home, away, delta)
```

### Logging Rules

* INFO: workflow milestones (loading, training, evaluation)
* DEBUG: internal states, values, calculations
* WARNING: recoverable issues (missing data, cache miss)
* ERROR: fatal errors

All logs should be timestamped and written to both console and file (`logs/experiment_<id>.log`).

---

## 8. Data Handling

### General Rules

* **Never mutate input DataFrames in place.**
* **Always copy** before transforming:

  ```python
  df = df.copy()
  ```
* Always label derived columns with clear names:

  * ✅ `elo_pre`, `elo_post`, `predicted_prob`
  * ❌ `x1`, `tmp`, `prob1`

### Vector Calculations

* Must respect `through_week` to avoid leakage.
* Document which columns are required inputs and which are derived.

---

## 9. Experiment Tracking

### SQLite Schema

* `experiments` table → metadata (name, date, config hash, summary metrics)
* `results` table → per-experiment aggregates
* `predictions` table → optional link to CSV path

### Naming

* Experiments should be timestamped:

  ```
  experiments/results/2025-11-02T13-30-elo_baseline.csv
  ```
* Every run gets an **experiment ID** (auto-increment).

---

## 10. Performance & Caching

### Caching Rules

* EloCache keyed by `(season, week, config_hash)`
* Only cache **finalized weekly states**, not partial updates
* Do not cache vector calculations unless reproducible deterministically

---

## 11. Testing & Validation

Use `pytest` with the following conventions:

* Unit tests for:

  * Elo rating updates (`test_elo.py`)
  * Vector calculations (`test_vectors.py`)
  * Metrics correctness (`test_metrics.py`)
* Integration tests for:

  * Full prediction flow
  * Experiment logging and retrieval

**Naming:**
`test_<module>.py` → test file
`test_<function>_<condition>()` → individual tests

Example:

```python
def test_elo_update_increases_winner_rating():
    ...
```

---

## 12. Code Quality Rules

| Rule                                       | Example                                            |
| ------------------------------------------ | -------------------------------------------------- |
| ✅ **Type hints everywhere**                | `def predict(self, home: str, away: str) -> dict:` |
| ✅ Limit functions to ≤ 40 lines            | Split complex logic                                |
| ✅ Prefer composition over inheritance      | VectorEnhancedPredictor uses EloPredictor instance |
| ✅ Use constants for repeated magic numbers | `HOME_ADVANTAGE_DEFAULT = 40`                      |
| ❌ No hard-coded paths                      | Use `Path` from `pathlib`                          |
| ❌ No global variables                      | Config + arguments only                            |
| ❌ No print()                               | Always use `logger`                                |

---

## 13. Code Style Enforcement

Use:

* **Black** (`line-length=100`)
* **isort** for import sorting
* **Flake8** for linting
* **mypy** for static type checks

Example `pyproject.toml` snippet:

```toml
[tool.black]
line-length = 100
target-version = ["py313"]

[tool.isort]
profile = "black"

[tool.flake8]
max-line-length = 100
ignore = ["E203", "W503"]

[tool.mypy]
ignore_missing_imports = true
```

---

## 14. Documentation Style

Each module (`.py` file) should start with a **1–3 line docstring**:

```python
"""
Elo rating model for NFL game predictions.

Implements an EloPredictor class compatible with BasePredictor interface.
Supports caching via EloCache.
"""
```

Every class should have a short summary docstring explaining its role and relationships to other modules.

---

## 15. Example Commit Messages

Use **Conventional Commits** style:

| Type        | Purpose               | Example                                                  |
| ----------- | --------------------- | -------------------------------------------------------- |
| `feat:`     | New feature           | `feat: add EloCache for faster week-by-week evaluations` |
| `fix:`      | Bug fix               | `fix: prevent through_week leakage in vector calc`       |
| `refactor:` | Internal change       | `refactor: move config parsing to utils/config.py`       |
| `test:`     | Add or improve tests  | `test: add calibration error validation test`            |
| `docs:`     | Documentation updates | `docs: update README with experiment runner usage`       |

---

## 16. Example Code Snippet (Putting It Together)

```python
from src.models.base import BasePredictor
from src.utils.cache import EloCache
from src.utils.logging import get_logger

logger = get_logger(__name__)

class EloPredictor(BasePredictor):
    """Elo rating predictor for NFL games."""

    def __init__(self, config: dict):
        self.config = config
        self.k_factor = config.get("k_factor", 20)
        self.home_advantage = config.get("home_advantage", 40)
        self.ratings = {}
        self.cache = EloCache()

    def fit(self, season: int, through_week: int = None):
        """Fit Elo ratings for a season up to a given week."""
        cached = self.cache.get(season, through_week, self.config)
        if cached:
            logger.info(f"Loaded Elo cache for season {season}, week {through_week}")
            self.ratings = cached
            return self
        # Compute Elo ratings ...
        self.cache.save(season, through_week, self.config, self.ratings)
        return self

    def predict(self, home_team: str, away_team: str) -> dict:
        """Predict win probability using current Elo ratings."""
        # compute win probability...
        return {"home_prob": 0.63}
```
