# NFL Elo Visualizations Guide

## Quick Start

```bash
# Generate all standard plots (saved to data/plots/)
python scripts/visualize.py

# Compare two k-factors visually
python scripts/visualize.py --compare 20 30

# Quick interactive preview
python scripts/visualize.py --quick --show
```

## What Gets Generated

### Standard Plots (`python scripts/visualize.py`)

1. **`elo_evolution_top8_k20.png`** - How top 8 teams' ratings changed over season
   - See who improved, who declined
   - Spot momentum shifts

2. **`elo_evolution_all_k20.png`** - All 32 teams over time
   - Messy but shows full picture
   - Good for finding sleeper teams

3. **`prediction_accuracy_k20.png`** - How well predictions matched reality
   - Left: Scatter of all predictions
   - Right: Calibration curve (are 70% predictions actually 70%?)

4. **`strength_of_schedule_k20.png`** - Who had it toughest
   - Red bars = harder than average schedule
   - Green bars = easier than average

5. **`biggest_upsets_k20.png`** - Most surprising results
   - Higher bar = bigger upset

### Comparison Plots (`python scripts/visualize.py --compare 20 30`)

1. **`ranking_comparison_k20_vs_k30.png`** - Side-by-side rankings
   - Left: Rating comparison bars
   - Right: Biggest movers between k-factors
   - Shows which teams are volatile vs stable

2. **`kfactor_heatmap_k20_vs_k30.png`** - Rank changes as heatmap
   - Green = higher rank (better)
   - Red = lower rank (worse)
   - Quickly see who moves where

3. **`dashboard_k20_vs_k30.png`** - Complete comparison dashboard
   - 6 panels showing everything at once
   - Best for presentations or reports

## Command Options

```bash
# Basic options
--season YEAR      # Which season (default: 2024)
--k NUMBER         # K-factor (default: 20)
--home NUMBER      # Home advantage (default: 65)

# Comparison
--compare K1 K2    # Compare two k-factors

# Display options
--show             # Display plots interactively (don't just save)
--nosave           # Don't save to disk (show only)
--quick            # Quick interactive mode (3 key plots)
```

## Usage Examples

### Example 1: Standard Analysis
```bash
# Generate plots for 2024 with K=20
python scripts/visualize.py

# Files created in data/plots/
ls data/plots/
```

### Example 2: Different K-Factor
```bash
# Try K=30 to see more volatility
python scripts/visualize.py --k 30

# Compare outputs
open data/plots/elo_evolution_top8_k20.png
open data/plots/elo_evolution_top8_k30.png
```

### Example 3: Compare K-Factors
```bash
# Generate comparison visualizations
python scripts/visualize.py --compare 20 30

# Open the dashboard
open data/plots/dashboard_k20_vs_k30.png
```

### Example 4: Interactive Exploration
```bash
# Quick look (shows plots on screen)
python scripts/visualize.py --quick --show

# Full analysis, view interactively
python scripts/visualize.py --show
```

### Example 5: Historical Analysis
```bash
# Compare 2023 vs 2024 with same K-factor
python scripts/visualize.py --season 2023 --k 25
python scripts/visualize.py --season 2024 --k 25

# Compare the top teams evolution
```

## Reading the Visualizations

### Elo Evolution Plots
- **Y-axis**: Elo rating (1500 = average)
- **X-axis**: Week of season
- **Gray dashed line**: League average (1500)
- **Steep climbs**: Team getting hot
- **Steep drops**: Team collapsing

**What to look for:**
- Teams that start low but trend up (momentum)
- Teams with stable high ratings (consistent)
- Wild swings (inconsistent or schedule-dependent)

### Prediction Accuracy Plots
- **Left plot**: Each dot is a game
  - On red line = perfect prediction
  - Above line = overconfident (predicted win, lost)
  - Below line = underconfident

- **Right plot**: Calibration curve
  - On red line = perfectly calibrated
  - Above = model is pessimistic
  - Below = model is optimistic

**Good calibration:**
- When you predict 70%, team wins ~70% of the time
- Curve stays close to diagonal line

### Strength of Schedule
- **Left side (red bars)**: Faced tough opponents (>1500 avg)
- **Right side (green bars)**: Faced weak opponents (<1500 avg)
- **Black dashed line**: League average

**Insights:**
- Teams with hard schedules might be underrated by W-L record
- Teams with easy schedules might be overrated

### Comparison Dashboard (6 panels)
**Top row**: Elo evolution for both k-factors
- Compare volatility (K=40 has more swings)

**Middle row**: Ranking bars
- Shows who moves up/down between k-factors

**Bottom left**: SOS comparison
- Usually similar (schedule doesn't change)

**Bottom right**: Accuracy comparison  
- Which k-factor predicts better?

## Tips for Analysis

### Finding Overrated Teams
1. Check SOS (easy schedule?)
2. Check Elo evolution (lucky win streak?)
3. Compare K=20 vs K=40 (drops with higher K = unsustainable)

### Finding Underrated Teams
1. Hard SOS but decent Elo
2. Elo higher than W-L rank suggests
3. Stable rating despite losses (losing close games)

### Finding Upset Potential
1. Teams trending up (Elo evolution climbing)
2. Favorable matchup (good run game vs weak run D - we'll add this next!)
3. Team underrated by Vegas/public (we'll compare later)

## Next Steps

Once you've generated these, we can:
1. Add matchup-specific visualizations (your "vectors" idea)
2. Create animated evolution (show week-by-week changes)
3. Add playoff probability predictions
4. Compare to Vegas betting lines

## Troubleshooting

**"No module named matplotlib"**
```bash
pip install matplotlib seaborn
```

**Plots look weird/small**
- Saved plots are high-res (300 DPI)
- If viewing on screen, they might look odd
- Open the PNG files directly for best quality

**Want different colors/styles?**
- Edit `src/analysis/visualizations.py`
- Change `sns.set_style()` options
- Modify color schemes in individual plot functions
