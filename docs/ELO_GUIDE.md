# Elo Rating System - Quick Start Guide

## What We Built

A comprehensive Elo rating system for NFL teams that:
- ✅ Accounts for margin of victory
- ✅ Includes home field advantage
- ✅ Tracks historical ratings
- ✅ Predicts game outcomes
- ✅ Identifies upsets and over/underperforming teams

## Files Created

1. **`src/models/elo.py`** - Core Elo rating engine
2. **`src/analysis/rankings.py`** - Analysis tools (compare to records, strength of schedule)

## Quick Run

```bash
# Make sure you're in the project directory with venv activated
cd ~/Repos/nfl-ratings
source venv/bin/activate

# Run Elo ratings for 2024 season
python -m src.models.elo
```

**You should see:**
- Current Elo rankings (1-32)
- Biggest upsets of the season
- Example prediction (Chiefs vs Bills)

## Run Analysis

```bash
# Compare Elo rankings to actual records
python -m src.analysis.rankings
```

**You'll get:**
- Full comparison of Elo rank vs Win-Loss rank
- Most overrated/underrated teams
- Strength of schedule analysis

## Using It Programmatically

```python
from src.models.elo import EloRatingSystem

# Initialize
elo = EloRatingSystem(
    k_factor=20,           # How much ratings change (higher = more volatile)
    home_advantage=65,     # Point advantage for home teams
    mov_multiplier=True    # Scale by margin of victory
)

# Calculate ratings
elo.calculate_season(2024, save_to_db=True)

# Get rankings
rankings = elo.get_rankings()
print(rankings)

# Predict a game
prediction = elo.predict_game('KC', 'BUF')  # Chiefs at home vs Bills
print(f"Chiefs win probability: {prediction['home_win_probability']:.1%}")

# Find upsets
upsets = elo.get_biggest_upsets(10)
print(upsets)
```

## What Makes a Team "Better" by Elo?

Unlike win-loss records, Elo considers:

1. **Who you beat** - Beating good teams increases rating more
2. **How much you won by** - Blowouts matter more than close games
3. **Expected outcome** - Upsets cause bigger rating swings
4. **Home field advantage** - Adjusted in calculations

## Key Insights You Can Find

### 1. Quality Wins
A team with a 9-8 record might rank higher than a 10-7 team if they beat better opponents.

### 2. Lucky/Unlucky Teams
Teams with close game records (lots of 1-score wins) might be overrated by record.

### 3. Strength of Schedule
Elo inherently adjusts for who you played, not just your record.

### 4. True Team Quality
Strips away luck and focuses on expected performance.

## Tuning Parameters

Want to experiment? Change these:

```python
# More volatile ratings (reacts faster to new results)
elo = EloRatingSystem(k_factor=30)

# Less home field advantage
elo = EloRatingSystem(home_advantage=50)

# Ignore margin of victory (only W/L matters)
elo = EloRatingSystem(mov_multiplier=False)
```

## Next Steps

1. **Run it!** See the 2024 rankings
2. **Compare** to actual standings - who's over/underrated?
3. **Historical analysis** - Load 2023 data and compare
4. **Predictions** - Predict upcoming games
5. **Tune parameters** - Experiment with k-factor and home advantage

## Example Questions You Can Answer

- "Who are the best teams who missed the playoffs?"
- "Which playoff team is the weakest?"
- "Who had the toughest schedule?"
- "What were the biggest upsets?"
- "Which team improved the most throughout the season?"

Have fun! 🏈
