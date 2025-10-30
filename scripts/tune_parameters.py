"""Compare different Elo parameters to find optimal settings."""
import pandas as pd
import numpy as np
from src.models.elo import EloRatingSystem
from src.database.db_manager import DatabaseManager


def evaluate_elo_parameters(season, k_factors, home_advantages):
    """
    Test multiple k-factor and home advantage combinations.
    
    Evaluates each on:
    - Prediction accuracy
    - Brier score (calibration)
    - Log loss
    
    Returns DataFrame with results.
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


def find_best_parameters(season=2024):
    """Find optimal k-factor and home advantage."""
    print("="*70)
    print("ELO PARAMETER TUNING")
    print("="*70)
    print(f"Testing different k-factors and home advantages on {season} season")
    print("This will take a few minutes...\n")
    
    # Test ranges
    k_factors = [10, 15, 20, 25, 30, 40]
    home_advantages = [40, 50, 60, 65, 70, 80]
    
    results = evaluate_elo_parameters(season, k_factors, home_advantages)
    
    # Display results
    print("\n" + "="*70)
    print("RESULTS (Top 10 by Accuracy)")
    print("="*70)
    print(results.head(10).to_string(index=False))
    
    print("\n" + "="*70)
    print("RESULTS (Top 10 by Brier Score - calibration)")
    print("="*70)
    print(results.sort_values('brier_score').head(10).to_string(index=False))
    
    # Best overall
    best = results.iloc[0]
    print("\n" + "="*70)
    print("RECOMMENDED PARAMETERS")
    print("="*70)
    print(f"K-Factor: {int(best['k_factor'])}")
    print(f"Home Advantage: {int(best['home_advantage'])}")
    print(f"Accuracy: {best['accuracy']:.2%}")
    print(f"Brier Score: {best['brier_score']:.4f}")
    print(f"Correct: {int(best['correct_predictions'])}/{int(best['total_games'])}")
    
    # Save results
    results.to_csv('data/processed/parameter_tuning_results.csv', index=False)
    print(f"\nFull results saved to: data/processed/parameter_tuning_results.csv")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Tune Elo parameters')
    parser.add_argument('--season', type=int, default=2024, help='Season to tune on')
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer parameters')
    
    args = parser.parse_args()
    
    if args.quick:
        # Quick test
        print("Running quick test...")
        k_factors = [15, 20, 25]
        home_advantages = [60, 65, 70]
        results = evaluate_elo_parameters(args.season, k_factors, home_advantages)
        print("\n" + "="*70)
        print("QUICK TEST RESULTS")
        print("="*70)
        print(results.to_string(index=False))
    else:
        # Full tuning
        results = find_best_parameters(args.season)
