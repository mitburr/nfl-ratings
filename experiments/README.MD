# Experiment Framework

Config-driven system for reproducible model evaluation.

## Philosophy

**Problem:** Script sprawl. Testing new ideas required new scripts with duplicated logic.

**Solution:** One runner, many configs. Define experiments declaratively, execute uniformly.

## Directory Structure
```
experiments/
├── configs/              # Experiment definitions (YAML)
│   ├── elo_baseline.yaml
│   ├── vectors_cumulative.yaml
│   └── vectors_weighted.yaml
├── results/              # Output directory
│   ├── experiments.db    # SQLite tracking
│   └── {exp_id}_{season}.csv  # Detailed predictions
├── runner.py             # Main execution engine
└── tracker.py            # Experiment tracking
```

## Config File Anatomy
```yaml
# experiments/configs/my_experiment.yaml
name: "descriptive_name"           # Unique identifier
model_type: "elo" | "vectors"      # Which model class to use

parameters:                        # Passed to model constructor
  k_factor: 20
  home_advantage: 40
  # For vectors:
  elo_config:                      # Nested config for base Elo
    k_factor: 20
    home_advantage: 40
  vector_config:                   # Vector-specific params
    boost: 0.60
    weight: 0.35
    blend_prior: true
    prior_weight: 0.25

metrics:                           # Which metrics to compute
  - accuracy
  - log_loss
  - brier_score
  - calibration_error
```

## Running Experiments

### Basic Execution
```bash
# Single experiment
python -m experiments.runner \
  --config experiments/configs/elo_baseline.yaml \
  --seasons 2024

# Multiple seasons
python -m experiments.runner \
  --config experiments/configs/elo_baseline.yaml \
  --seasons 2022 2023 2024
```

### CLI Overrides

Modify config params without editing files:
```bash
# Single param
python -m experiments.runner \
  --config experiments/configs/elo_baseline.yaml \
  --seasons 2024 \
  --override k_factor=25

# Multiple params
python -m experiments.runner \
  --config experiments/configs/vectors_weighted.yaml \
  --seasons 2024 \
  --override vector_config.boost=0.7 vector_config.weight=0.4

# Nested params use dot notation
python -m experiments.runner \
  --config experiments/configs/vectors_weighted.yaml \
  --seasons 2024 \
  --override elo_config.k_factor=25 vector_config.blend_prior=false
```

### Comparison Mode

Compare multiple configs in one run:
```bash
# Compare all configs
python -m experiments.runner \
  --compare experiments/configs/*.yaml \
  --seasons 2024

# Compare specific configs
python -m experiments.runner \
  --compare \
    experiments/configs/elo_baseline.yaml \
    experiments/configs/vectors_cumulative.yaml \
    experiments/configs/vectors_weighted.yaml \
  --seasons 2024

# Output: CSV with side-by-side comparison
```

## Execution Flow
```
1. Load config(s) → YAML parser
2. Apply overrides → Merge CLI params
3. Validate config → Check required fields, types
4. Create model → Instantiate predictor class
5. For each season:
   a. Fit model (rolling=True)
   b. Predict all games
   c. Compute metrics
   d. Save predictions (optional)
6. Log to database → SQLite tracker
7. Save summary → CSV output
```

### Rolling Evaluation (Critical)

**What it does:**
- Refits model before each week
- Uses only games through (week - 1) for predictions
- Mimics real-world prediction scenario

**Implementation:**
```python
current_week = None
for game in season_games:
    if game.week != current_week:
        model.fit(season, through_week=game.week)  # Refit!
        current_week = game.week
    
    prediction = model.predict(game.home, game.away)
    # Compare to actual result
```

**Why it matters:**
Prevents data leakage. Without rolling evaluation:
- Model sees future games during training
- Artificially inflates accuracy
- Predictions unrealistic

**Performance cost:**
- ~18 fits per season (one per week)
- Mitigated by EloCache (caches intermediate states)

**Disable with:** `--no-rolling`
- Faster, but results invalid for publication
- Use only for quick sanity checks

## Experiment Tracking

All runs logged to `experiments/results/experiments.db`

**Schema:**
```sql
experiments
  - id (auto-increment)
  - name (from config)
  - timestamp
  - config (JSON)
  - seasons (JSON list)
  - avg_accuracy, avg_log_loss, avg_brier_score
  - total_games

season_results
  - experiment_id (FK)
  - season
  - accuracy, log_loss, brier_score
  - total_games
  - predictions_path (CSV link)
```

### Querying History
```python
from experiments.tracker import ExperimentTracker

tracker = ExperimentTracker()

# List recent experiments
history = tracker.get_experiments(limit=20)
print(history[['id', 'name', 'avg_accuracy', 'timestamp']])

# Get details for specific experiment
details = tracker.get_experiment_details(experiment_id=42)
print(details['season_results'])
print(details['experiment']['config'])

# Compare experiments
comparison = tracker.compare_experiments([40, 41, 42])
print(comparison)
```

### CLI Query Tool
```bash
# View history
python -c "from experiments.tracker import ExperimentTracker; \
           t = ExperimentTracker(); \
           print(t.get_experiments())"

# Or add to Taskfile:
task history
```

## Creating New Configs

### Template
```yaml
name: "my_new_experiment"
model_type: "elo"  # or "vectors"
parameters:
  # Required for all models
  k_factor: 20
  home_advantage: 40
  initial_rating: 1500
  mov_multiplier: true
  
  # For vector models, add:
  # elo_config:
  #   k_factor: 20
  #   home_advantage: 40
  # vector_config:
  #   boost: 0.60
  #   weight: 0.35
  #   blend_prior: false

metrics:
  - accuracy
  - log_loss
  - brier_score
```

### Validation

Configs are validated on load. Common errors:

**Missing required field:**
```yaml
name: "my_exp"
# ERROR: Missing model_type and parameters
```

**Invalid model_type:**
```yaml
model_type: "random_forest"  # ERROR: Not in ['elo', 'vectors']
```

**Invalid parameter type:**
```yaml
parameters:
  k_factor: "twenty"  # ERROR: Must be number
```

**Invalid nested parameter:**
```yaml
parameters:
  vector_config:
    weight: 1.5  # ERROR: Must be in [0, 1]
```

## Parameter Sweeps

Test multiple values of a parameter:

**Manual approach:**
```bash
for k in 15 20 25 30; do
  python -m experiments.runner \
    --config configs/elo_baseline.yaml \
    --seasons 2024 \
    --override k_factor=$k
done
```

**Better: Loop in script**
```bash
#!/bin/bash
for k in 15 20 25 30; do
  for ha in 30 40 50; do
    python -m experiments.runner \
      --config configs/elo_baseline.yaml \
      --seasons 2024 \
      --override k_factor=$k home_advantage=$ha
  done
done
```

**Best: Grid search config**
```yaml
# configs/k_factor_sweep.yaml
name: "k_factor_sweep"
model_type: "elo"
sweep:
  parameters: ["k_factor"]
  values: [15, 20, 25, 30]
base_parameters:
  home_advantage: 40
  initial_rating: 1500
```

Then create custom runner script or use existing runner in loop.

## Output Locations

**Logs:**
- `logs/experiment_{name}_{timestamp}.log`
- Contains full execution trace
- DEBUG level for troubleshooting

**Predictions:**
- `experiments/results/{exp_id}_{season}.csv`
- Per-game predictions with metadata
- Columns: week, home_team, away_team, predicted_prob, actual_result, correct, metadata

**Summary:**
- `experiments/results/comparison_{timestamp}.csv`
- Aggregated metrics across experiments
- Generated by `--compare` mode

**Database:**
- `experiments/results/experiments.db`
- Persistent record of all runs
- Query with `ExperimentTracker`

## Advanced Usage

### Custom Metrics

Add new metric to registry:
```python
# src/evaluation/metrics.py
def custom_metric(predictions: pd.DataFrame) -> float:
    # Calculate metric
    return value

METRICS['custom_metric'] = custom_metric
```

Then use in config:
```yaml
metrics:
  - accuracy
  - custom_metric
```

### Multi-Season Aggregation

Compare performance across seasons:
```python
from experiments.tracker import ExperimentTracker
import pandas as pd

tracker = ExperimentTracker()

# Get all experiments matching pattern
experiments = tracker.get_experiments(name_pattern='vector')

# Aggregate by model type
for _, exp in experiments.iterrows():
    details = tracker.get_experiment_details(exp['id'])
    seasons = details['season_results']
    
    print(f"\n{exp['name']}")
    print(seasons[['season', 'accuracy', 'log_loss']])
    print(f"Average: {seasons['accuracy'].mean():.3f}")
```

### Debugging Failed Experiments
```bash
# Check log file
tail -f logs/experiment_vectors_20251102_143022.log

# Run with verbose output
python -m experiments.runner \
  --config configs/problematic.yaml \
  --seasons 2024 \
  # Add to runner.py: setup_logging('DEBUG')

# Validate config independently
python -c "from src.utils.config import load_config, validate_config; \
           validate_config(load_config('configs/problematic.yaml'))"
```

## Performance Tips

**Cache is King:**
- EloCache stores intermediate states
- Massive speedup for repeated fits
- Automatically used when config unchanged

**Reduce Rolling Granularity:**
- Instead of refitting every week, refit every 2-3 weeks
- Slight accuracy drop, major speed gain
- Requires code modification in evaluator

**Parallel Execution:**
- Run different configs in separate terminals
- Or: Add multiprocessing to runner.py
- Database locking handled by SQLite

**Sample Data:**
- Test on single season first
- Then expand to multi-season
- Use `--seasons 2024` for quick iteration

## Common Patterns

### Weekly Prediction Workflow
```bash
# Monday: Load latest results
task load-current

# Tuesday: Retrain all models
for config in experiments/configs/*.yaml; do
  python -m experiments.runner --config $config --seasons 2025
done

# Wednesday: Generate predictions for upcoming week
python -m scripts.predict_week --week 10 --config configs/best_model.yaml

# Sunday: Compare predictions to actual results
# (Manual analysis or add evaluation script)
```

### Research Iteration
```bash
# 1. Hypothesis: Higher k_factor improves accuracy
python -m experiments.runner \
  --config configs/elo_baseline.yaml \
  --seasons 2022 2023 2024 \
  --override k_factor=30

# 2. Compare to baseline
python -m experiments.runner --compare \
  configs/elo_baseline.yaml \
  configs/elo_k30.yaml \
  --seasons 2022 2023 2024

# 3. Analyze results
python -c "from experiments.tracker import ExperimentTracker; \
           t = ExperimentTracker(); \
           print(t.compare_experiments([last_two_ids]))"

# 4. If promising, create dedicated config
# configs/elo_k30.yaml
```

### Model Development Cycle
```
1. Implement new model in src/models/
2. Create minimal config in experiments/configs/
3. Test on single season: --seasons 2024
4. Compare to baseline: --compare
5. If better, run on all seasons: --seasons 2022 2023 2024
6. Tune parameters with --override
7. Create final optimized config
8. Add to production rotation
```

## Troubleshooting

**"Config file not found"**
- Check path is relative to project root
- Use: `experiments/configs/name.yaml` (not `configs/name.yaml`)

**"Model not fitted" warning**
- Model fit failed, check logs
- Likely: no data for specified season/week
- Verify: `task db-check`

**"Database locked" error**
- Another experiment running
- SQLite doesn't handle concurrent writes well
- Wait or kill other process

**Predictions seem identical to baseline**
- Check vector calculations ran (see logs)
- Verify `blend_prior` setting correct
- Check `boost` and `weight` aren't zero

**Results not reproducible**
- Ensure same season and through_week
- Check random seeds if using any
- Verify database hasn't changed

## Next Steps

**Add new model:**
1. Create `src/models/my_model.py` (extend BasePredictor)
2. Register in `experiments/runner.py` → `create_model()`
3. Create config `experiments/configs/my_model.yaml`
4. Test: `python -m experiments.runner --config configs/my_model.yaml --seasons 2024`

**Optimize existing model:**
1. Create sweep configs with different params
2. Run: `python -m experiments.runner --compare configs/model_*.yaml --seasons 2024`
3. Analyze: `tracker.compare_experiments([...])`
4. Update production config with best params

**Deploy predictions:**
1. Identify best model from history
2. Add to weekly Taskfile routine
3. Automate with cron or CI/CD
4. Monitor accuracy over time