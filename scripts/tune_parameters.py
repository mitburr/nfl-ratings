"""Compare different Elo parameters to find optimal settings across all historical data."""
import pandas as pd
import numpy as np
from src.models.elo import EloRatingSystem
from src.database.db_manager import DatabaseManager


def evaluate_elo_parameters_historical(seasons, k_factors, home_advantages, regression_factor=0.33):
    """
    Test multiple k-factor and home advantage combinations across all historical data.
    
    Uses continuous Elo ratings with between-season regression toward the mean.
    This simulates real-world usage where ratings carry over between seasons.
    
    Args:
        seasons: List of seasons to evaluate (e.g., [2010, 2011, ..., 2024])
        k_factors: List of k-factors to test
        home_advantages: List of home advantages to test
        regression_factor: Amount to regress toward mean between seasons (default 0.33 = 1/3)
    
    Evaluates each on:
    - Prediction accuracy
    - Brier score (calibration)
    - Log loss
    
    Returns DataFrame with results.
    """
    results = []
    
    db = DatabaseManager()
    
    # Get all completed games across all seasons, sorted chronologically
    seasons_str = ', '.join([str(s) for s in seasons])
    query = f"""
        SELECT game_id, season, week, game_date, home_team, away_team, home_score, away_score
        FROM games
        WHERE season IN ({seasons_str}) AND home_score IS NOT NULL
        ORDER BY game_date, game_id
    """
    games = db.query_to_dataframe(query)
    
    if len(games) == 0:
        print("ERROR: No games found for specified seasons")
        return pd.DataFrame()
    
    print(f"Loaded {len(games)} games across {len(seasons)} seasons")
    print(f"Date range: {games['game_date'].min()} to {games['game_date'].max()}")
    
    total_combinations = len(k_factors) * len(home_advantages)
    current = 0
    
    for k in k_factors:
        for ha in home_advantages:
            current += 1
            print(f"\n[{current}/{total_combinations}] Testing K={k}, Home={ha}...")
            
            # Initialize Elo with these parameters
            elo = EloRatingSystem(k_factor=k, home_advantage=ha)
            
            # Track predictions vs actuals
            predictions = []
            actuals = []
            correct = 0
            total = 0
            brier_scores = []
            
            # Track current season for regression
            current_season = None
            season_ratings = {}  # Store end-of-season ratings
            
            # Process each game chronologically
            for idx, game in games.iterrows():
                # Check if we've moved to a new season
                if current_season is not None and game['season'] != current_season:
                    # Apply regression toward mean (1500) for all teams
                    print(f"  Season boundary: {current_season} → {game['season']} (applying {regression_factor:.0%} regression)")
                    for team in elo.ratings:
                        old_rating = elo.ratings[team]
                        new_rating = (1 - regression_factor) * old_rating + regression_factor * 1500
                        elo.ratings[team] = new_rating
                
                current_season = game['season']
                
                # Make prediction before updating ratings
                pred = elo.predict_game(game['home_team'], game['away_team'])
                home_win_prob = pred['home_win_probability']
                
                # Actual outcome
                home_won = 1 if game['home_score'] > game['away_score'] else 0
                
                predictions.append(home_win_prob)
                actuals.append(home_won)
                
                # Check if prediction was correct
                predicted_winner = 'home' if home_win_prob > 0.5 else 'away'
                actual_winner = 'home' if home_won else 'away'
                
                if predicted_winner == actual_winner:
                    correct += 1
                total += 1
                
                # Brier score for this prediction
                brier_scores.append((home_win_prob - home_won) ** 2)
                
                # Update ratings for next prediction
                elo.update_ratings(
                    game['home_team'],
                    game['away_team'],
                    game['home_score'],
                    game['away_score'],
                    game['week'],
                    game['season'],
                    game['game_id']
                )
            
            # Calculate metrics
            accuracy = correct / total
            brier_score = np.mean(brier_scores)
            
            # Log loss (lower is better)
            log_loss = -np.mean([
                actual * np.log(max(pred, 0.001)) + (1 - actual) * np.log(max(1 - pred, 0.001))
                for pred, actual in zip(predictions, actuals)
            ])
            
            results.append({
                'k_factor': k,
                'home_advantage': ha,
                'accuracy': accuracy,
                'brier_score': brier_score,
                'log_loss': log_loss,
                'correct_predictions': correct,
                'total_games': total
            })
            
            print(f"  Accuracy: {accuracy:.2%} ({correct}/{total}), Brier: {brier_score:.4f}, Log Loss: {log_loss:.4f}")
    
    results_df = pd.DataFrame(results)
    return results_df.sort_values('accuracy', ascending=False)


def find_best_parameters_historical(start_year=2010, end_year=2024, exclude_years=None):
    """
    Find optimal k-factor and home advantage using all historical data.
    
    Args:
        start_year: First season to include
        end_year: Last season to include
        exclude_years: List of years to exclude (e.g., [2020] for COVID season)
    """
    print("="*70)
    print("ELO PARAMETER TUNING - HISTORICAL DATA")
    print("="*70)
    
    # Build season list
    seasons = list(range(start_year, end_year + 1))
    if exclude_years:
        seasons = [s for s in seasons if s not in exclude_years]
        print(f"Testing across {len(seasons)} seasons: {start_year}-{end_year} (excluding {exclude_years})")
    else:
        print(f"Testing across {len(seasons)} seasons: {start_year}-{end_year}")
    
    print("Ratings will carry over between seasons with 1/3 regression toward mean")
    print("This simulates real-world usage of the Elo system")
    print("\nThis will take several minutes...\n")
    
    # Test ranges
    k_factors = [10, 15, 20, 25, 30, 35, 40]
    home_advantages = [40, 50, 55, 60, 65, 70, 80]
    
    results = evaluate_elo_parameters_historical(seasons, k_factors, home_advantages)
    
    if len(results) == 0:
        print("\nERROR: No results generated. Check that data is loaded.")
        print("Try running: python -m scripts.load_historical_data 2010 2024")
        return None
    
    # Display results
    print("\n" + "="*70)
    print("RESULTS (Top 10 by Accuracy)")
    print("="*70)
    print(results.head(10).to_string(index=False))
    
    print("\n" + "="*70)
    print("RESULTS (Top 10 by Brier Score - calibration)")
    print("="*70)
    print(results.sort_values('brier_score').head(10).to_string(index=False))
    
    print("\n" + "="*70)
    print("RESULTS (Top 10 by Log Loss)")
    print("="*70)
    print(results.sort_values('log_loss').head(10).to_string(index=False))
    
    # Best overall
    best = results.iloc[0]
    print("\n" + "="*70)
    print("RECOMMENDED PARAMETERS")
    print("="*70)
    print(f"K-Factor: {int(best['k_factor'])}")
    print(f"Home Advantage: {int(best['home_advantage'])}")
    print(f"Accuracy: {best['accuracy']:.2%}")
    print(f"Brier Score: {best['brier_score']:.4f}")
    print(f"Log Loss: {best['log_loss']:.4f}")
    print(f"Correct: {int(best['correct_predictions'])}/{int(best['total_games'])} games")
    
    # Save results
    results.to_csv('data/processed/parameter_tuning_historical_results.csv', index=False)
    print(f"\nFull results saved to: data/processed/parameter_tuning_historical_results.csv")
    
    return results


def evaluate_single_season(season, k_factors, home_advantages):
    """
    Evaluate parameters on a single season (legacy function for comparison).
    
    This resets Elo to 1500 at the start of the season.
    Useful for seeing how parameters perform on individual seasons.
    """
    results = []
    
    db = DatabaseManager()
    
    # Get all completed games for evaluation
    query = """
        SELECT game_id, week, home_team, away_team, home_score, away_score
        FROM games
        WHERE season = %s AND home_score IS NOT NULL
        ORDER BY week, game_date
    """
    games = db.query_to_dataframe(query, params=(season,))
    
    total_combinations = len(k_factors) * len(home_advantages)
    current = 0
    
    for k in k_factors:
        for ha in home_advantages:
            current += 1
            print(f"\n[{current}/{total_combinations}] Testing K={k}, Home={ha}...")
            
            # Initialize Elo with these parameters
            elo = EloRatingSystem(k_factor=k, home_advantage=ha)
            
            # Track predictions vs actuals
            predictions = []
            actuals = []
            correct = 0
            total = 0
            brier_scores = []
            
            # Process each game and make predictions
            for idx, game in games.iterrows():
                # Make prediction before updating ratings
                pred = elo.predict_game(game['home_team'], game['away_team'])
                home_win_prob = pred['home_win_probability']
                
                # Actual outcome
                home_won = 1 if game['home_score'] > game['away_score'] else 0
                
                predictions.append(home_win_prob)
                actuals.append(home_won)
                
                # Check if prediction was correct
                predicted_winner = 'home' if home_win_prob > 0.5 else 'away'
                actual_winner = 'home' if home_won else 'away'
                
                if predicted_winner == actual_winner:
                    correct += 1
                total += 1
                
                # Brier score for this prediction
                brier_scores.append((home_win_prob - home_won) ** 2)
                
                # Update ratings for next prediction
                elo.update_ratings(
                    game['home_team'],
                    game['away_team'],
                    game['home_score'],
                    game['away_score'],
                    game['week'],
                    season,
                    game['game_id']
                )
            
            # Calculate metrics
            accuracy = correct / total
            brier_score = np.mean(brier_scores)
            
            # Log loss (lower is better)
            log_loss = -np.mean([
                actual * np.log(max(pred, 0.001)) + (1 - actual) * np.log(max(1 - pred, 0.001))
                for pred, actual in zip(predictions, actuals)
            ])
            
            results.append({
                'k_factor': k,
                'home_advantage': ha,
                'accuracy': accuracy,
                'brier_score': brier_score,
                'log_loss': log_loss,
                'correct_predictions': correct,
                'total_games': total
            })
            
            print(f"  Accuracy: {accuracy:.1%}, Brier: {brier_score:.4f}, Log Loss: {log_loss:.4f}")
    
    results_df = pd.DataFrame(results)
    return results_df.sort_values('accuracy', ascending=False)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Tune Elo parameters')
    parser.add_argument('--season', type=int, help='Single season to tune on (legacy mode)')
    parser.add_argument('--start', type=int, default=2010, help='Start year for historical analysis')
    parser.add_argument('--end', type=int, default=2024, help='End year for historical analysis')
    parser.add_argument('--exclude', nargs='+', type=int, help='Years to exclude (e.g., --exclude 2020)')
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer parameters')
    
    args = parser.parse_args()
    
    if args.season:
        # Legacy mode: single season
        print(f"Running single-season analysis for {args.season}...")
        if args.quick:
            k_factors = [15, 20, 25]
            home_advantages = [60, 65, 70]
        else:
            k_factors = [10, 15, 20, 25, 30, 40]
            home_advantages = [40, 50, 60, 65, 70, 80]
        
        results = evaluate_single_season(args.season, k_factors, home_advantages)
        print("\n" + "="*70)
        print(f"SINGLE SEASON RESULTS ({args.season})")
        print("="*70)
        print(results.to_string(index=False))
    else:
        # Historical mode (default)
        exclude_years = args.exclude if args.exclude else [2020]
        results = find_best_parameters_historical(args.start, args.end, exclude_years)
