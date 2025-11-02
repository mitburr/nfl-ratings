# Source Code Architecture

Core prediction system with clean abstraction layers.

## Directory Structure
```
src/
├── models/              # Prediction models
│   ├── base.py          # BasePredictor interface
│   ├── elo.py           # Elo rating system
│   └── vector_enhanced.py  # Elo + vectors
├── evaluation/          # Model evaluation framework
│   ├── evaluator.py     # Cross-validation logic
│   └── metrics.py       # Performance metrics registry
├── analysis/            # Data analysis utilities
│   ├── team_vectors.py  # Play-by-play aggregations
│   ├── rankings.py      # Team ranking analysis
│   └── visualizations.py  # Plotting utilities
├── database/            # Database interface
│   ├── db_manager.py    # PostgreSQL connection
│   └── schema.sql       # Database schema
├── ingestion/           # Data loading
│   └── nfl_data.py      # nflverse data fetcher
└── utils/               # Shared utilities
    ├── config.py        # YAML config handling
    ├── cache.py         # Elo caching for performance
    └── logging.py       # Logging configuration
```

## Core Abstraction Layers

### Layer 1: Data (database/, ingestion/)

**Purpose:** Load NFL data from nflverse into PostgreSQL

**Key Classes:**
- `DatabaseManager`: Query interface with connection pooling
- `NFLDataIngestion`: Fetch schedules, play-by-play data

**Data Flow:**
```
nflverse API → NFLDataIngestion → PostgreSQL
                                       ↓
                                  DatabaseManager queries
```

**Common Operations:**
```python
from src.database.db_manager import DatabaseManager

db = DatabaseManager()
games = db.query_to_dataframe("SELECT * FROM games WHERE season = %s", params=(2024,))
```

---

### Layer 2: Models (models/)

**Purpose:** Prediction algorithms with consistent interface

**Base Interface:**
```python
class BasePredictor(ABC):
    def fit(self, season: int, through_week: int = None):
        """Train on games through specified week."""
        pass
    
    def predict(self, home_team: str, away_team: str) -> dict:
        """Return {home_win_prob, away_win_prob, spread, confidence, metadata}."""
        pass
```

**Implemented Models:**

**EloPredictor** (`elo.py`)
- Dynamic team ratings updated after each game
- Accounts for home advantage, margin of victory
- Caches ratings for performance (via `EloCache`)

**VectorEnhancedPredictor** (`vector_enhanced.py`)
- Wraps EloPredictor (composition pattern)
- Calculates team offensive/defensive efficiency vectors
- Adjusts Elo predictions based on matchup advantages

**Model Interaction:**
```
VectorEnhancedPredictor
    ├── EloPredictor (composition)
    │   └── EloCache
    └── team_vectors.py (for adjustments)
```

**Key Design Decisions:**

1. **Composition over inheritance**: VectorEnhanced *has* an EloPredictor, doesn't extend it
   - Easier to swap/test components
   - Clearer separation of concerns

2. **Config-driven**: Models configured via dicts, not constructor args
   - Enables experiment framework
   - Easy serialization/comparison

3. **Stateful fit/predict**: Models store fitted state
   - `fit()` loads/processes data
   - `predict()` uses fitted state
   - Enables caching and incremental updates

**Adding a New Model:**
```python
# src/models/my_model.py
from src.models.base import BasePredictor

class MyPredictor(BasePredictor):
    def __init__(self, config: dict):
        super().__init__(config)
        # Extract config params
        self.my_param = config.get('my_param', default_value)
    
    def fit(self, season: int, through_week: int = None):
        # Load data via DatabaseManager
        # Calculate model parameters
        self.is_fitted = True
        return self
    
    def predict(self, home_team: str, away_team: str) -> dict:
        # Use fitted parameters to predict
        return {
            'home_win_prob': 0.6,
            'away_win_prob': 0.4,
            'spread': 3.5,
            'confidence': 0.8,
            'metadata': {'my_info': 'value'}
        }
```

Then register in `experiments/runner.py`:
```python
def create_model(config: dict):
    if config['model_type'] == 'my_model':
        return MyPredictor(config['parameters'])
```

---

### Layer 3: Evaluation (evaluation/)

**Purpose:** Test model accuracy with proper cross-validation

**Key Classes:**

**ModelEvaluator** (`evaluator.py`)
- Rolling evaluation: refits model each week
- Prevents data leakage
- Computes multiple metrics per season

**Metrics Registry** (`metrics.py`)
- Pluggable metric functions
- Standard: accuracy, log_loss, brier_score, calibration_error

**Evaluation Flow:**
```
ModelEvaluator.evaluate_season(model, season=2024, rolling=True)
    ↓
For each week:
    1. Fit model on games through (week - 1)
    2. Predict games in current week
    3. Record actual results
    ↓
Compute metrics on all predictions
```

**Critical:** `rolling=True` ensures no data leakage
- Model only sees past games when predicting
- Mimics real prediction scenario

**Adding a New Metric:**
```python
# evaluation/metrics.py
def roi_metric(predictions: pd.DataFrame) -> float:
    """Calculate return on investment if betting on favorites."""
    # Implementation
    return roi_value

METRICS['roi'] = roi_metric
```

---

### Layer 4: Analysis (analysis/)

**Purpose:** Data exploration and feature engineering

**team_vectors.py**
- Aggregates play-by-play data into team metrics
- **Critical:** Uses `through_week` parameter to prevent leakage
- Returns DataFrame with offensive/defensive stats

**Common Pattern:**
```python
from src.analysis.team_vectors import calculate_team_vectors

# For analysis (all games)
vectors = calculate_team_vectors(season=2024)

# For prediction (only prior games)
vectors = calculate_team_vectors(season=2024, through_week=8)
```

**Key Metrics:**
- `rush_ypa`, `pass_ypa` (offense)
- `rush_ypa_allowed`, `pass_ypa_allowed` (defense)
- `success_rate`, `explosive_rate`

**rankings.py**, **visualizations.py**
- Exploratory analysis tools
- Not used in prediction pipeline
- Useful for hypothesis testing

---

## Data Flow for Prediction
```
User request: "Predict Week 9 2025"
    ↓
scripts/predict_week.py
    ↓
Load config → create_model(config)
    ↓
EloPredictor.fit(2025, through_week=8)
    ↓
DatabaseManager.query_to_dataframe("SELECT ... WHERE week <= 8")
    ↓
Process games → update Elo ratings
    ↓
Cache final state (EloCache)
    ↓
For each Week 9 game:
    EloPredictor.predict(home, away)
    ↓
    Return {home_win_prob, spread, ...}
```

**With Vector Enhancement:**
```
VectorEnhancedPredictor.fit(2025, through_week=8)
    ├── EloPredictor.fit(2025, through_week=8)  [as above]
    └── calculate_team_vectors(2025, through_week=8)
        ↓
        Query plays table → aggregate metrics
        ↓
        Return team efficiency DataFrame

VectorEnhancedPredictor.predict(home, away)
    ├── elo_pred = EloPredictor.predict(home, away)
    ├── matchup = analyze_matchup(home, away, vectors)
    └── adjusted_prob = elo_pred + (matchup * boost * weight)
```

---

## Performance Considerations

**EloCache** (`utils/cache.py`)
- In-memory cache of Elo ratings by (season, week, config_hash)
- Avoids refitting when config unchanged
- Massive speedup for parameter sweeps

**When Cache Helps:**
- Running same experiment multiple times
- Comparing models with identical Elo config
- Parameter sweeps over vector params only

**When Cache Doesn't Help:**
- First run of any config
- Changing k_factor or home_advantage

**Memory Usage:**
- Each cached state: ~50KB (32 teams × ratings + history)
- Typical usage: 10-20 cached states = 1MB
- Safe to cache aggressively

---

## Common Development Patterns

### Adding a Feature to Existing Model

**Example:** Add time-decay to Elo ratings

1. **Update model class:**
```python
# src/models/elo.py
class EloPredictor(BasePredictor):
    def __init__(self, config: dict):
        # ...
        self.decay_rate = config.get('decay_rate', 0.0)  # New param
    
    def fit(self, season: int, through_week: int = None):
        # ... existing code ...
        # Apply decay between seasons
        if self.decay_rate > 0:
            self.ratings = {t: r * (1 - self.decay_rate) 
                           for t, r in self.ratings.items()}
```

2. **Update config:**
```yaml
# experiments/configs/elo_decay.yaml
name: "elo_with_decay"
model_type: "elo"
parameters:
  k_factor: 20
  home_advantage: 40
  decay_rate: 0.1  # New parameter
```

3. **Test:**
```bash
python -m experiments.runner --config configs/elo_decay.yaml --seasons 2024
```

### Creating a New Feature Type

**Example:** Add "momentum" based on recent performance

1. **Create feature calculator:**
```python
# src/analysis/momentum.py
def calculate_momentum(season: int, through_week: int) -> pd.DataFrame:
    """Calculate team momentum from recent games."""
    db = DatabaseManager()
    # Query recent game results
    # Calculate win streak, point differential trend, etc.
    return momentum_df
```

2. **Integrate into model:**
```python
# src/models/momentum_enhanced.py
from src.models.elo import EloPredictor
from src.analysis.momentum import calculate_momentum

class MomentumEnhancedPredictor(BasePredictor):
    def __init__(self, config: dict):
        super().__init__(config)
        self.elo = EloPredictor(config['elo_config'])
        self.momentum_weight = config.get('momentum_weight', 0.2)
    
    def fit(self, season: int, through_week: int = None):
        self.elo.fit(season, through_week)
        self.momentum = calculate_momentum(season, through_week)
        self.is_fitted = True
        return self
    
    def predict(self, home_team: str, away_team: str) -> dict:
        elo_pred = self.elo.predict(home_team, away_team)
        momentum_adj = self._calculate_momentum_adjustment(home_team, away_team)
        adjusted_prob = elo_pred['home_win_prob'] + momentum_adj
        return {
            'home_win_prob': adjusted_prob,
            'away_win_prob': 1 - adjusted_prob,
            'spread': (adjusted_prob - 0.5) * 28,
            'confidence': abs(adjusted_prob - 0.5) * 2,
            'metadata': {**elo_pred['metadata'], 'momentum_adj': momentum_adj}
        }
```

3. **Register and test:**
```python
# experiments/runner.py
def create_model(config):
    if config['model_type'] == 'momentum':
        return MomentumEnhancedPredictor(config['parameters'])
```

---

## Testing Strategy

**Unit Tests** (`tests/`)
- Test individual model components
- Mock database calls
- Focus on calculation correctness

**Integration Tests**
- Test full prediction pipeline
- Use real database (test subset)
- Verify predictions match expected format

**Evaluation Tests**
- Ensure no data leakage
- Verify metrics computed correctly
- Check rolling evaluation works

**Example Test:**
```python
# tests/test_elo.py
def test_elo_update_increases_winner():
    config = {'k_factor': 20, 'home_advantage': 40}
    elo = EloPredictor(config)
    
    initial_winner = elo.get_rating('KC')
    elo.update_ratings('KC', 'BUF', home_score=30, away_score=20, 
                       week=1, season=2024)
    final_winner = elo.get_rating('KC')
    
    assert final_winner > initial_winner
```

---

## Debugging Tips

**Model not fitting properly?**
- Check `through_week` parameter (common leakage source)
- Verify database has games for that week
- Check cache - clear with `elo.cache.clear()` if stale

**Predictions seem wrong?**
- Verify `is_fitted` is True
- Check metadata in prediction dict
- Compare Elo-only vs enhanced predictions

**Performance issues?**
- Profile with `cProfile`
- Check if cache is being used (log level DEBUG)
- Consider reducing rolling evaluation granularity

**Config not working?**
- Validate config with `validate_config()`
- Check for typos in nested keys
- Use `--override` to test parameter changes

---

## Future Architecture Considerations

**If adding 5+ models:**
- Implement ModelPipeline abstraction
- Chain transformations: Base → Adjustment₁ → Adjustment₂
- Cleaner than deep nesting

**If models become stateful across seasons:**
- Implement proper serialization in `get_state()`/`load_state()`
- Store model checkpoints in database
- Enable warm starts

**If adding real-time updates:**
- Stream game results via WebSocket
- Incremental Elo updates (don't refit entire season)
- Pub/sub for prediction updates

**If scaling to 100+ experiments:**
- Move to proper experiment framework (MLflow)
- Distributed evaluation (Dask/Ray)
- Cloud storage for results