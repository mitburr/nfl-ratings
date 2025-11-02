"""
Weekly NFL Predictions using project infrastructure.

Usage:
    python -m scripts.predict_week --week 9 --season 2025
    python -m scripts.predict_week  # Defaults to current week
"""
import argparse
from datetime import datetime
import pandas as pd
from src.models.elo import EloRatingSystem
from src.database.db_manager import DatabaseManager


def get_upcoming_games(season, week):
    """Get games scheduled for a specific week."""
    db = DatabaseManager()
    query = """
        SELECT game_id, week, game_date, home_team, away_team
        FROM games
        WHERE season = %s AND week = %s
        ORDER BY game_date
    """
    return db.query_to_dataframe(query, params=(season, week))


def predict_week(season, week):
    """Generate predictions for an upcoming week."""
    print("="*70)
    print(f"NFL WEEK {week} PREDICTIONS - {datetime.now().strftime('%B %d, %Y')}")
    print("="*70)
    print()
    
    # Calculate current Elo ratings (through previous week)
    print(f"Calculating Elo ratings through Week {week-1}...")
    elo = EloRatingSystem(k_factor=20, home_advantage=40)
    elo.calculate_season(season, through_week=week-1, save_to_db=False)
    print(f"✓ Processed {len(elo.history)} games")
    print(f"✓ Current ratings for {len(elo.ratings)} teams\n")
    
    # Show top 10 teams
    print("="*70)
    print(f"TOP 10 TEAMS (After Week {week-1})")
    print("="*70)
    rankings = pd.DataFrame([
        {'team': team, 'rating': rating}
        for team, rating in elo.ratings.items()
    ]).sort_values('rating', ascending=False).head(10)
    rankings['rank'] = range(1, len(rankings) + 1)
    print(rankings[['rank', 'team', 'rating']].to_string(index=False))
    print()
    
    # Get upcoming games
    games = get_upcoming_games(season, week)
    
    if len(games) == 0:
        print(f"⚠ No games found for Week {week}")
        return None
    
    print("="*70)
    print(f"WEEK {week} GAME PREDICTIONS")
    print("="*70)
    print()
    
    predictions = []
    
    for _, game in games.iterrows():
        home = game['home_team']
        away = game['away_team']
        date = game['game_date']
        
        pred = elo.predict_game(home, away)
        
        home_prob = pred['home_win_probability']
        away_prob = pred['away_win_probability']
        spread = pred['spread_estimate']
        favorite = pred['favorite']
        fav_prob = home_prob if favorite == home else away_prob
        
        predictions.append({
            'date': date,
            'away_team': away,
            'home_team': home,
            'favorite': favorite,
            'fav_prob': fav_prob,
            'home_prob': home_prob,
            'away_prob': away_prob,
            'spread': spread,
            'home_elo': pred['home_rating'],
            'away_elo': pred['away_rating']
        })
        
        # Print each game
        print(f"📅 {date}")
        print(f"   {away:>4} @ {home:<4}")
        print(f"   {home}: {home_prob:.1%} (Elo {pred['home_rating']:.0f})")
        print(f"   {away}: {away_prob:.1%} (Elo {pred['away_rating']:.0f})")
        print(f"   ⭐ Favorite: {favorite} ({fav_prob:.1%})")
        print(f"   📊 Spread: {home} {spread:+.1f}")
        print()
    
    # Summary table
    print("="*70)
    print("SUMMARY")
    print("="*70)
    df = pd.DataFrame(predictions).sort_values('fav_prob', ascending=False)
    for _, row in df.iterrows():
        matchup = f"{row['away_team']} @ {row['home_team']}"
        print(f"{matchup:15} | {row['favorite']} favored {row['fav_prob']:.1%} | Spread: {row['spread']:+.1f}")
    print("="*70)
    
    # Save to CSV
    filename = f'data/processed/week{week}_predictions_{season}.csv'
    df.to_csv(filename, index=False)
    print(f"\n✓ Saved: {filename}")
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict upcoming NFL games')
    parser.add_argument('--week', type=int, help='Week number (default: current week)')
    parser.add_argument('--season', type=int, default=2025, help='Season year')
    
    args = parser.parse_args()
    
    # If no week specified, try to detect current week
    if args.week is None:
        # Simple heuristic: it's November, probably Week 9
        db = DatabaseManager()
        query = """
            SELECT MAX(week) as last_completed
            FROM games
            WHERE season = %s AND home_score IS NOT NULL
        """
        result = db.query_to_dataframe(query, params=(args.season,))
        last_completed = result['last_completed'].iloc[0]
        args.week = last_completed + 1 if last_completed else 1
        print(f"Auto-detected: Predicting Week {args.week}\n")
    
    try:
        predict_week(args.season, args.week)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("  1. Database is set up (python -m scripts.setup)")
        print("  2. Season data is loaded (python -m scripts.load_historical_data 2025)")
        print("  3. Database config is correct (config/database.py)")
