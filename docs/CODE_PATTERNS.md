# Code Patterns Quick Reference

## When Creating New Scripts...

### ✅ DO THIS (Integrate with project)

```python
"""scripts/my_new_script.py - Proper integration"""
from src.models.elo import EloRatingSystem
from src.database.db_manager import DatabaseManager
from src.analysis.team_vectors import calculate_team_vectors

def main():
    db = DatabaseManager()  # Uses config/database.py
    elo = EloRatingSystem()  # Existing class
    vectors = calculate_team_vectors(2024, through_week=8)
    # ... your new logic here

if __name__ == "__main__":
    main()
```

### ❌ DON'T DO THIS (Standalone duplication)

```python
"""my_script.py - ANTI-PATTERN"""
import psycopg2

class EloRatingSystem:  # ❌ Duplicating existing code
    pass

def get_db_connection():
    return psycopg2.connect(  # ❌ Not using DatabaseManager
        database='nfl_ratings',
        password='hardcoded'  # ❌ Hardcoded credentials
    )
```

---

## Common Imports

```python
# Elo predictions
from src.models.elo import EloRatingSystem

# Database queries
from src.database.db_manager import DatabaseManager

# Team vectors
from src.analysis.team_vectors import (
    calculate_team_vectors,
    analyze_matchup
)

# Data loading
from src.ingestion.nfl_data import NFLDataIngestion

# Visualizations
from src.analysis.visualizations import (
    plot_elo_history,
    plot_team_comparison
)
```

---

## Checklist for New Scripts

- [ ] Imports from `src/` modules (not standalone)
- [ ] Uses `DatabaseManager` (not raw psycopg2)
- [ ] No hardcoded credentials
- [ ] Follows `scripts/` naming convention
- [ ] Includes docstring explaining purpose
- [ ] Uses `through_week` parameter if calculating vectors in-season
- [ ] Saves output to `data/processed/` or `data/plots/`

---

## Common Patterns

### Making Predictions
```python
elo = EloRatingSystem(k_factor=20, home_advantage=40)
elo.calculate_season(2024, through_week=8)
pred = elo.predict_game('KC', 'BUF')
print(f"KC: {pred['home_win_probability']:.1%}")
```

### Querying Database
```python
db = DatabaseManager()
query = "SELECT * FROM games WHERE season = %s"
games = db.query_to_dataframe(query, params=(2024,))
```

### Calculating Vectors
```python
# Full season (for analysis)
vectors = calculate_team_vectors(2024)

# Through specific week (for predictions)
vectors = calculate_team_vectors(2024, through_week=8)

# Matchup analysis
matchup = analyze_matchup('KC', 'BUF', vectors)
```

### Loading Data
```python
ingestion = NFLDataIngestion()
ingestion.fetch_schedule([2024, 2025])
ingestion.fetch_play_by_play([2024])
```

---

## Red Flags 🚩

If you see these in new code, it's wrong:

- `class EloRatingSystem:` in a script (should import it)
- `psycopg2.connect()` anywhere (should use DatabaseManager)
- `password='...'` in code (should use config)
- Calculating vectors without `through_week` for in-season predictions
- Scripts not importing from `src/`

---

## File Locations

**Scripts:** `scripts/*.py`
- Import from `src/`
- User-facing commands
- Can be run with `python -m scripts.script_name`

**Source Code:** `src/`
- Core functionality
- Reusable modules
- Never run directly

**Results:** `data/`
- `data/processed/*.csv` - Analysis results
- `data/plots/*.png` - Visualizations
- `data/raw/` - Raw data (usually empty, data comes from nflreadpy)

**Config:** `config/database.py`
- Database credentials (never hardcode!)
- Read via DatabaseManager

---

## When User Says "Help me predict this week's games"

```python
# ✅ Create: scripts/predict_week.py
from src.models.elo import EloRatingSystem
from src.database.db_manager import DatabaseManager

def predict_week(season, week):
    elo = EloRatingSystem()
    elo.calculate_season(season, through_week=week-1)
    
    db = DatabaseManager()
    games = db.query_to_dataframe(
        "SELECT * FROM games WHERE season=%s AND week=%s",
        params=(season, week)
    )
    
    for _, game in games.iterrows():
        pred = elo.predict_game(game['home_team'], game['away_team'])
        print(f"{game['away_team']} @ {game['home_team']}: {pred['home_win_probability']:.1%}")

# ❌ DON'T create standalone script with embedded Elo class
```

---

## Remember

**The project already has:**
- Elo system (`src/models/elo.py`)
- Database interface (`src/database/db_manager.py`)
- Vector calculations (`src/analysis/team_vectors.py`)
- Data loading (`src/ingestion/nfl_data.py`)

**Your job is to:**
- Use these existing components
- Add new scripts that import from them
- Never duplicate functionality
- Follow the established patterns