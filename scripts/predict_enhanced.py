"""Predict NFL games using Elo + team vectors."""
import argparse
import pandas as pd
from src.models.elo import EloRatingSystem
from src.database.db_manager import DatabaseManager
from src.analysis.team_vectors import calculate_team_vectors, analyze_matchup


def calculate_current_elo(season=2025, k_factor=20, home_advantage=40, verbose=True):
    """Calculate current Elo ratings for a season."""
    if verbose:
        print(f"Calculating Elo ratings for {season}...")
        print(f"Parameters: K={k_factor}, Home={home_advantage}")
    
    elo = EloRatingSystem(k_factor=k_factor, home_advantage=home_advantage)
    elo.calculate_season(season, save_to_db=False)
    
    if verbose:
        print(f"✓ Processed {len(elo.history)} games")
        print(f"✓ Rated {len(elo.ratings)} teams\n")
    
    return elo


def predict_with_vectors(elo, home_team, away_team, vectors_df, vector_weight=0.15):
    """
    Predict game using Elo + vector matchup adjustment.
    
    Args:
        elo: EloRatingSystem with current ratings
        home_team: Home team abbreviation
        away_team: Away team abbreviation  
        vectors_df: Team vectors DataFrame
        vector_weight: Weight for vector adjustment (0-1, default 0.15)
    
    Returns:
        dict with prediction details including vector boost
    """
    # Get base Elo prediction
    base_pred = elo.predict_game(home_team, away_team)
    
    # Get vector matchup analysis
    matchup = analyze_matchup(home_team, away_team, vectors_df)
    
    if matchup is None:
        # No vector data, return base Elo
        return {**base_pred, 'vector_adjustment': 0, 'method': 'elo_only'}
    
    # Vector advantage is in yards/play difference
    # Normalize: +1 yard advantage = ~10% win probability boost
    vector_boost = matchup['total_advantage'] * 0.1
    
    # Apply weight to vector boost
    adjusted_boost = vector_boost * vector_weight
    
    # Adjust probabilities
    new_home_prob = base_pred['home_win_probability'] + adjusted_boost
    new_home_prob = max(0.01, min(0.99, new_home_prob))  # Clamp to [0.01, 0.99]
    new_away_prob = 1 - new_home_prob
    
    # Recalculate spread
    rating_diff = new_home_prob - 0.5
    new_spread = rating_diff * 32  # Rough conversion
    
    return {
        'home_team': home_team,
        'away_team': away_team,
        'home_rating': base_pred['home_rating'],
        'away_rating': base_pred['away_rating'],
        'elo_home_prob': base_pred['home_win_probability'],
        'vector_adjustment': adjusted_boost,
        'home_win_probability': new_home_prob,
        'away_win_probability': new_away_prob,
        'spread_estimate': new_spread,
        'favorite': home_team if new_home_prob > 0.5 else away_team,
        'method': 'elo+vectors',
        'vector_matchup': matchup
    }


def evaluate_predictions(season=2024, use_vectors=True, vector_weight=0.15, 
                        k_factor=20, home_advantage=40):
    """
    Evaluate prediction accuracy on completed games.
    
    Returns:
        DataFrame with prediction results
    """
    print("="*70)
    print(f"EVALUATING PREDICTIONS FOR {season}")
    print("="*70)
    print(f"Method: {'Elo + Vectors' if use_vectors else 'Elo Only'}")
    print(f"Parameters: K={k_factor}, Home={home_advantage}")
    if use_vectors:
        print(f"Vector weight: {vector_weight}")
    print()
    
    # Calculate Elo ratings
    elo = EloRatingSystem(k_factor=k_factor, home_advantage=home_advantage)
    
    # Get vectors if using them
    vectors_df = calculate_team_vectors(season) if use_vectors else None
    
    # Get all completed games
    db = DatabaseManager()
    query = """
        SELECT game_id, week, home_team, away_team, home_score, away_score
        FROM games
        WHERE season = %s AND home_score IS NOT NULL
        ORDER BY week, game_date
    """
    games = db.query_to_dataframe(query, params=(season,))
    
    results = []
    correct = 0
    total = 0
    
    for idx, game in games.iterrows():
        # Make prediction
        if use_vectors and vectors_df is not None:
            pred = predict_with_vectors(elo, game['home_team'], game['away_team'], 
                                       vectors_df, vector_weight)
        else:
            pred = elo.predict_game(game['home_team'], game['away_team'])
        
        # Actual outcome
        home_won = game['home_score'] > game['away_score']
        predicted_home = pred['home_win_probability'] > 0.5
        
        correct += (home_won == predicted_home)
        total += 1
        
        results.append({
            'game_id': game['game_id'],
            'week': game['week'],
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'predicted_winner': game['home_team'] if predicted_home else game['away_team'],
            'actual_winner': game['home_team'] if home_won else game['away_team'],
            'correct': home_won == predicted_home,
            'home_prob': pred['home_win_probability'],
            'home_score': game['home_score'],
            'away_score': game['away_score']
        })
        
        # Update Elo for next prediction
        elo.update_ratings(
            game['home_team'], game['away_team'],
            game['home_score'], game['away_score'],
            game['week'], season, game['game_id']
        )
    
    accuracy = correct / total
    
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total} games)")
    print("="*70)
    
    return pd.DataFrame(results), accuracy


def format_prediction(pred, home_team, away_team):
    """Format prediction for display."""
    print("="*70)
    print("GAME PREDICTION")
    print("="*70)
    print(f"{away_team} @ {home_team}")
    print()
    
    if pred.get('method') == 'elo+vectors':
        print(f"Elo ratings:")
        print(f"  {home_team}: {pred['home_rating']:.1f}")
        print(f"  {away_team}: {pred['away_rating']:.1f}")
        print()
        print(f"Elo prediction: {home_team} {pred['elo_home_prob']:.1%}")
        print(f"Vector adjustment: {pred['vector_adjustment']:+.1%}")
        print()
        print(f"Final prediction:")
        print(f"  {home_team}: {pred['home_win_probability']:.1%}")
        print(f"  {away_team}: {pred['away_win_probability']:.1%}")
    else:
        print(f"{home_team}: Elo {pred['home_rating']:.1f} - {pred['home_win_probability']:.1%}")
        print(f"{away_team}: Elo {pred['away_rating']:.1f} - {pred['away_win_probability']:.1%}")
    
    print()
    print(f"Favorite: {pred['favorite']}")
    print(f"Spread: {home_team} {pred['spread_estimate']:+.1f}")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict NFL games with Elo + Vectors')
    parser.add_argument('--season', type=int, default=2025)
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--home-adv', type=int, default=40)
    parser.add_argument('--predict', nargs=2, metavar=('AWAY', 'HOME'))
    parser.add_argument('--evaluate', action='store_true', help='Evaluate on season')
    parser.add_argument('--no-vectors', action='store_true', help='Use Elo only')
    parser.add_argument('--vector-weight', type=float, default=0.15, 
                       help='Weight for vectors (0-1)')
    
    args = parser.parse_args()
    
    if args.evaluate:
        # Evaluate accuracy
        results, accuracy = evaluate_predictions(
            args.season, 
            use_vectors=not args.no_vectors,
            vector_weight=args.vector_weight,
            k_factor=args.k,
            home_advantage=args.home_adv
        )
        
        # Save results
        results.to_csv(f'data/processed/predictions_{args.season}.csv', index=False)
        print(f"\nResults saved to data/processed/predictions_{args.season}.csv")
        
    elif args.predict:
        # Single prediction
        elo = calculate_current_elo(args.season, args.k, args.home_adv)
        
        if not args.no_vectors:
            vectors = calculate_team_vectors(args.season)
            pred = predict_with_vectors(
                elo, args.predict[1].upper(), args.predict[0].upper(),
                vectors, args.vector_weight
            )
        else:
            pred = elo.predict_game(args.predict[1].upper(), args.predict[0].upper())
        
        format_prediction(pred, args.predict[1].upper(), args.predict[0].upper())
    
    else:
        print("Use --predict AWAY HOME or --evaluate")