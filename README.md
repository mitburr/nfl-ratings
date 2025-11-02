# NFL Prediction System

Statistical prediction system for NFL games using Elo ratings and advanced team performance metrics.

## Current Status

**Production Models:**
- **Elo Baseline**: 64-65% accuracy using dynamic team ratings
- **Vector-Enhanced**: Elo + play-by-play efficiency metrics (experimental)

**Latest Research:** Testing weighted vs. cumulative vector approaches for consistent multi-season performance.

## Quick Start
```bash
# Load current season data
task load-current

# Run experiment
task experiment -- --config experiments/configs/elo_baseline.yaml --seasons 2024

# Predict upcoming week
task predict-week -- 9

# Compare models
task compare -- 2024
```

## Architecture
```
├── src/                    # Core prediction models and evaluation
├── experiments/            # Config-driven experiment framework
├── scripts/                # Data loading and prediction utilities
├── data/
│   ├── processed/          # Prediction outputs
│   └── experiments/        # Experiment results
└── logs/                   # Experiment logs
```

## Key Features

**Config-Driven Experiments:** Define models in YAML, run via single command
```bash
python -m experiments.runner --config configs/elo_baseline.yaml --seasons 2024
```

**CLI Overrides:** Tune parameters without editing configs
```bash
python -m experiments.runner --config configs/elo_baseline.yaml --override k_factor=25
```

**Experiment Tracking:** SQLite database tracks all runs
```python
from experiments.tracker import ExperimentTracker
tracker = ExperimentTracker()
history = tracker.get_experiments()
```

**Proper Evaluation:** Rolling cross-validation prevents data leakage
- Models refit each week using only prior games
- No future information in predictions

## Creating New Models

1. Extend `BasePredictor` in `src/models/`
2. Implement `fit()` and `predict()` methods
3. Create config in `experiments/configs/`
4. Run via experiment runner

See `src/README.md` for detailed architecture guide.

## Research Goals

1. Develop novel statistical methods for NFL team/player evaluation
2. Achieve accurate game predictions (>70% target)
3. Build reproducible experiment framework

## Tech Stack

- Python 3.13
- PostgreSQL (game/play data)
- SQLite (experiment tracking)
- pandas, numpy, scipy
- nflreadpy (data source)

## Documentation

- `src/README.md` - Architecture deep dive
- `experiments/README.md` - Experiment framework guide
- `docs/CODE_PATTERNS.md` - Code style guide
- `docs/SETUP_GUIDE.md` - Installation instructions

## Common Tasks
```bash
# Data
task load -- 2022 2024        # Load multiple seasons
task load-current             # Load 2025 season

# Experiments
task experiment -- --config configs/elo_baseline.yaml --seasons 2024
task compare -- 2024          # Compare all configs
task sweep -- k_factor=25     # Parameter sweep

# Predictions
task predict-week -- 9                    # Current config
task predict-elo -- 9                     # Elo only
task predict-vectors -- 9                 # Vector-enhanced

# Analysis
task analyze-vectors          # Exploratory analysis
task history                  # View past experiments
```

## Project State

**Working:**
- Elo baseline (stable 64-65% accuracy)
- Vector calculations (rush/pass efficiency)
- Experiment tracking & comparison
- Rolling evaluation framework

**In Progress:**
- Multi-season vector generalization
- Optimal prior season blending weight
- Calibration improvements

**Next Steps:**
- Situational metrics (3rd down, red zone)
- Ensemble methods
- Player-level data integration