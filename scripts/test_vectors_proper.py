"""Multi-season test of Elo vs vector-based approaches with proper rolling evaluation + statistical testing."""
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
from src.models.elo import EloRatingSystem
from src.database.db_manager import DatabaseManager
from src.analysis.team_vectors import calculate_team_vectors, analyze_matchup


# -----------------------------------------------------------------------------
# VECTOR BLENDING
# -----------------------------------------------------------------------------
def calculate_weighted_vectors(prior_season, current_season, current_week, prior_weight=0.25):
    pd.set_option('future.no_silent_downcasting', True)
    """Blend prior season (full) with current season (through current_week)."""
    prior = calculate_team_vectors(prior_season)
    current = calculate_team_vectors(current_season, through_week=current_week)

    merged = prior.merge(current, on='team', suffixes=('_prior', '_current'), how='outer')

    # Fill missing values safely (suppress FutureWarning)
    for suffix in ['_prior', '_current']:
        for col in merged.columns:
            if col.endswith(suffix):
                merged[col] = merged[col].fillna(merged[col].mean())
    merged = merged.infer_objects(copy=False)

    metrics = [
        'rush_ypa', 'pass_ypa', 'rush_ypa_allowed', 'pass_ypa_allowed',
        'success_rate', 'explosive_rate', 'success_rate_allowed', 'explosive_rate_allowed'
    ]

    for metric in metrics:
        prior_col = f'{metric}_prior'
        current_col = f'{metric}_current'
        if prior_col in merged.columns and current_col in merged.columns:
            merged[metric] = (
                merged[prior_col] * prior_weight +
                merged[current_col] * (1 - prior_weight)
            )

    final_cols = ['team'] + metrics
    return merged[[col for col in final_cols if col in merged.columns]]


# -----------------------------------------------------------------------------
# EVALUATION LOOP
# -----------------------------------------------------------------------------
def evaluate_season_rolling(test_season, approach, boost=0.60, weight=0.35):
    """
    Evaluate predictive accuracy with proper rolling (no leakage) for one season.
    Approaches:
        'elo'        → Elo only
        'cumulative' → Elo + current-season vectors
        'weighted'   → Elo + weighted (prior + current) vectors
    """
    elo = EloRatingSystem(k_factor=20, home_advantage=40)

    db = DatabaseManager()
    query = """
        SELECT game_id, week, home_team, away_team, home_score, away_score
        FROM games
        WHERE season = %s AND home_score IS NOT NULL
        ORDER BY week, game_date
    """
    games = db.query_to_dataframe(query, params=(test_season,))
    if len(games) == 0:
        print(f"No games found for {test_season}")
        return None

    correct_elo = 0
    correct_method = 0
    current_week = None
    vectors_df = None

    for _, game in games.iterrows():
        # Recalculate vectors when the week changes
        if approach != 'elo' and game['week'] != current_week:
            current_week = game['week']
            if approach == 'cumulative':
                vectors_df = calculate_team_vectors(test_season, through_week=current_week)
            elif approach == 'weighted':
                vectors_df = calculate_weighted_vectors(test_season - 1, test_season, current_week)

        # --- ELO prediction ---
        base_pred = elo.predict_game(game['home_team'], game['away_team'])
        new_home_prob = base_pred['home_win_probability']

        # --- Apply vector adjustment ---
        if approach != 'elo' and vectors_df is not None:
            matchup = analyze_matchup(game['home_team'], game['away_team'], vectors_df)
            if matchup:
                vector_boost = matchup['total_advantage'] * boost
                adjusted_boost = vector_boost * weight
                new_home_prob = max(0.01, min(0.99, new_home_prob + adjusted_boost))

        # --- Evaluate correctness ---
        home_won = game['home_score'] > game['away_score']
        correct_elo += (home_won == (base_pred['home_win_probability'] > 0.5))
        correct_method += (home_won == (new_home_prob > 0.5))

        # --- Update Elo after the game ---
        elo.update_ratings(
            game['home_team'], game['away_team'],
            game['home_score'], game['away_score'],
            game['week'], test_season, game['game_id']
        )

    n = len(games)
    return {
        'season': test_season,
        'approach': approach,
        'elo_accuracy': correct_elo / n,
        'method_accuracy': correct_method / n,
        'improvement': (correct_method - correct_elo) / n
    }


# -----------------------------------------------------------------------------
# MULTI-SEASON COMPARISON + STATS
# -----------------------------------------------------------------------------
def run_significance_tests(df):
    """Perform paired t-tests on season-level accuracies between approaches."""
    import pandas as pd
    from scipy.stats import ttest_rel
    import numpy as np

    # Adopt the future pandas behavior to silence the fillna downcasting warning
    pd.set_option('future.no_silent_downcasting', True)

    # Pivot for easier comparison
    pivot = df.pivot(index='season', columns='approach', values='method_accuracy')

    def safe_ttest(a, b):
        """Run a paired t-test safely, skipping NaN seasons."""
        mask = (~a.isna()) & (~b.isna())
        if mask.sum() < 3:
            return np.nan, np.nan
        return ttest_rel(a[mask], b[mask])

    pairs = [
        ('elo', 'cumulative'),
        ('elo', 'weighted'),
        ('cumulative', 'weighted')
    ]

    print("\nStatistical Significance (paired t-tests across seasons)")
    print("-" * 75)

    for a, b in pairs:
        if a in pivot.columns and b in pivot.columns:
            tstat, pval = safe_ttest(pivot[a], pivot[b])
            if np.isnan(pval):
                print(f"{a.upper()} vs {b.upper()}: insufficient data")
            else:
                sig = "SIGNIFICANT" if pval < 0.05 else "ns"
                print(f"{a.upper()} vs {b.upper()}: t={tstat:+.3f}, p={pval:.4f} → {sig}")


def multi_season_comparison():
    """Compare Elo, cumulative vectors, and weighted vectors across all seasons."""
    print("=" * 75)
    print("MULTI-SEASON COMPARISON: ELO vs VECTOR APPROACHES (ROLLING, NO LEAKAGE)")
    print("=" * 75)
    print("Parameters: Boost=0.60, Weight=0.35")
    print("Prior-season weight: 25%\n")

    db = DatabaseManager()
    query = """
        SELECT DISTINCT g.season 
        FROM plays p 
        JOIN games g ON p.game_id = g.game_id 
        ORDER BY g.season
    """
    available = db.query_to_dataframe(query)['season'].tolist()
    test_seasons = [s for s in available if s > min(available)]
    print(f"Testing seasons: {test_seasons}\n")

    results = []

    for season in test_seasons:
        print(f"Season {season}")
        print("-" * 60)

        elo_only = evaluate_season_rolling(season, 'elo')
        print(f"  ELO ONLY:    {elo_only['method_accuracy']:.2%}")
        results.append(elo_only)

        cum = evaluate_season_rolling(season, 'cumulative')
        print(f"  CUMULATIVE:  {cum['method_accuracy']:.2%} ({cum['improvement']:+.2%})")
        results.append(cum)

        wgt = evaluate_season_rolling(season, 'weighted')
        print(f"  WEIGHTED:    {wgt['method_accuracy']:.2%} ({wgt['improvement']:+.2%})")
        results.append(wgt)
        print()

    df = pd.DataFrame(results)

    # --- Summary ---
    print("=" * 75)
    print("SUMMARY ACROSS ALL SEASONS")
    print("=" * 75)

    avg_elo = df[df['approach'] == 'elo']['method_accuracy'].mean()
    avg_cum = df[df['approach'] == 'cumulative']['method_accuracy'].mean()
    avg_wgt = df[df['approach'] == 'weighted']['method_accuracy'].mean()

    cum_improve = df[df['approach'] == 'cumulative']['improvement'].mean()
    wgt_improve = df[df['approach'] == 'weighted']['improvement'].mean()

    print(f"ELO only avg accuracy:         {avg_elo:.2%}")
    print(f"Cumulative vector avg accuracy:{avg_cum:.2%} ({cum_improve:+.2%} vs Elo)")
    print(f"Weighted vector avg accuracy:  {avg_wgt:.2%} ({wgt_improve:+.2%} vs Elo)\n")

    best = max(avg_elo, avg_cum, avg_wgt)
    winner = (
        "ELO ONLY" if best == avg_elo else
        "CUMULATIVE" if best == avg_cum else
        "WEIGHTED"
    )
    print(f"Winner overall: {winner}")
    print("=" * 75)

    # --- Run t-tests ---
    run_significance_tests(df)

    return df


# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    results_df = multi_season_comparison()
    results_df.to_csv('data/processed/multi_season_elo_vector_comparison.csv', index=False)
    print("\nResults saved to: data/processed/multi_season_elo_vector_comparison.csv")
