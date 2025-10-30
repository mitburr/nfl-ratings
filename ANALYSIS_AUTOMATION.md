# Analysis Automation Scripts

## Quick Answer to Your Questions

**Q: Are the analysis tools idempotent?**  
**A: YES!** Each run starts fresh from 1500 ratings. You can test different k-factors without interference.

**Q: Can I run all analysis at once?**  
**A: YES!** Use the new `run_full_analysis.py` script.

---

## Scripts Overview

### 1. `run_full_analysis.py` - All-in-One Analysis

Runs complete analysis pipeline in one command:
- ✅ Calculate Elo ratings
- ✅ Show rankings
- ✅ Find biggest upsets
- ✅ Compare to W-L records
- ✅ Identify over/underrated teams
- ✅ Strength of schedule analysis

**Basic usage:**
```bash
python scripts/run_full_analysis.py
```

**With custom parameters:**
```bash
# Higher k-factor (more reactive)
python scripts/run_full_analysis.py --k 30

# Different home advantage
python scripts/run_full_analysis.py --home 70

# Different season
python scripts/run_full_analysis.py --season 2023

# Save to database
python scripts/run_full_analysis.py --save

# Debug specific team
python scripts/run_full_analysis.py --debug MIA

# Combine options
python scripts/run_full_analysis.py --k 25 --home 60 --save
```

**All options:**
- `--season YEAR` - Which season (default: 2024)
- `--k NUMBER` - K-factor 1-50 (default: 20)
- `--home NUMBER` - Home advantage in Elo points (default: 65)
- `--save` - Save results to database
- `--debug TEAM` - Show detailed logs for a team

---

### 2. `tune_parameters.py` - Find Optimal K-Factor

Tests multiple k-factor and home advantage combinations to find which predicts best.

**Metrics evaluated:**
- **Accuracy** - % of games predicted correctly
- **Brier Score** - How well-calibrated probabilities are (lower = better)
- **Log Loss** - Penalizes confident wrong predictions (lower = better)

**Full tuning (takes ~5 minutes):**
```bash
python scripts/tune_parameters.py
```

Tests:
- K-factors: 10, 15, 20, 25, 30, 40
- Home advantages: 40, 50, 60, 65, 70, 80
- Total: 36 combinations

**Quick test (~1 minute):**
```bash
python scripts/tune_parameters.py --quick
```

Tests only 9 combinations.

**Output:**
- Prints top 10 by accuracy
- Prints top 10 by Brier score
- Recommends best parameters
- Saves full results to `data/processed/parameter_tuning_results.csv`

---

## Example Workflows

### Workflow 1: Quick Analysis
```bash
# Run with default settings
python scripts/run_full_analysis.py

# See rankings, upsets, over/underrated teams, SOS
```

### Workflow 2: Compare K-Factors
```bash
# Test K=15
python scripts/run_full_analysis.py --k 15 > results_k15.txt

# Test K=20
python scripts/run_full_analysis.py --k 20 > results_k20.txt

# Test K=30
python scripts/run_full_analysis.py --k 30 > results_k30.txt

# Compare the rankings to see which feels right
```

### Workflow 3: Find Optimal Parameters
```bash
# Find best k-factor and home advantage
python scripts/tune_parameters.py

# Use recommended parameters
python scripts/run_full_analysis.py --k 25 --home 60 --save
```

### Workflow 4: Historical Analysis
```bash
# Analyze 2023 season
python scripts/run_full_analysis.py --season 2023

# Compare 2023 vs 2024 top teams
```

---

## Understanding the Metrics

### Accuracy
Simple: Did we predict the winner correctly?
- 50% = random guessing
- 65-70% = good model
- 75%+ = excellent (NFL is hard to predict!)

### Brier Score
Measures probability calibration:
- 0.0 = perfect calibration
- 0.25 = random guessing
- Lower is better
- Example: If you say 70% chance of winning 10 times, team should win ~7 times

### Log Loss
Penalizes confident wrong predictions heavily:
- Lower is better
- Being 90% confident and wrong is much worse than being 60% confident and wrong

---

## Tips

1. **K-factor sweet spot:** Usually 20-30 for NFL
   - Lower = more stable, favors historically good teams
   - Higher = more reactive, rewards recent performance

2. **Home advantage:** Around 60-70 points seems right
   - Equivalent to ~2.5 point spread in Vegas terms

3. **Don't over-optimize:** 
   - Best parameters on one season may not work next season
   - NFL has randomness - 70% accuracy is excellent

4. **Save results:**
   - Use `--save` only after you're happy with parameters
   - Database stores one set of ratings per week

---

## What's Next?

Now that you have automation, you can:
1. **Test different parameters** - Find what works best
2. **Add visualizations** - Plot Elo over time, strength of schedule
3. **Build upset predictor** - Use matchup-specific stats
4. **Historical analysis** - Load 2022-2023 data and compare

Ready to explore any of these?
