"""Predict NFL game outcomes using current Elo ratings."""
import argparse
from src.models.elo import EloRatingSystem
from src.database.db_manager import DatabaseManager


def calculate_current_elo(season=2025, k_factor=20, home_advantage=40, verbose=True):
    """
    Calculate current Elo ratings for a season.
    
    Args:
        season: Season to analyze
        k_factor: Elo k-factor (default: 20 from historical tuning)
        home_advantage: Home field advantage (default: 40 from historical tuning)
        verbose: Print progress
    
    Returns:
        EloRatingSystem with current ratings
    """
    if verbose:
        print(f"Calculating current Elo ratings for {season} season...")
        print(f"Using optimal parameters: K={k_factor}, Home={home_advantage}")
    
    # Initialize with optimal historical parameters
    elo = EloRatingSystem(k_factor=k_factor, home_advantage=home_advantage)
    
    # Calculate ratings from completed games
    elo.calculate_season(season, save_to_db=False)
    
    if verbose:
        print(f"✓ Processed {len(elo.history)} completed games")
        print(f"✓ Rated {len(elo.ratings)} teams\n")
    
    return elo


def predict_game(elo, home_team, away_team, neutral_site=False):
    """
    Predict outcome of a specific game.
    
    Args:
        elo: EloRatingSystem with current ratings
        home_team: Home team abbreviation (e.g., 'MIA')
        away_team: Away team abbreviation (e.g., 'BAL')
        neutral_site: Is this a neutral site game? (default: False)
    
    Returns:
        dict with prediction details
    """
    # Temporarily disable home advantage for neutral site
    original_home_adv = elo.home_advantage
    if neutral_site:
        elo.home_advantage = 0
    
    prediction = elo.predict_game(home_team, away_team)
    
    # Restore home advantage
    elo.home_advantage = original_home_adv
    
    return prediction


def format_prediction(pred, home_team, away_team):
    """Format prediction for display."""
    home_prob = pred['home_win_probability']
    away_prob = pred['away_win_probability']
    spread = pred['spread_estimate']
    
    print("="*60)
    print("GAME PREDICTION")
    print("="*60)
    print(f"{away_team} @ {home_team}")
    print()
    print(f"{home_team:>4} - Elo: {pred['home_rating']:.1f} - Win Prob: {home_prob:.1%}")
    print(f"{away_team:>4} - Elo: {pred['away_rating']:.1f} - Win Prob: {away_prob:.1%}")
    print()
    print(f"Favorite: {pred['favorite']}")
    print(f"Estimated Spread: {home_team} {spread:+.1f}")
    print("="*60)


def show_current_rankings(elo, top_n=10):
    """Display current top teams."""
    rankings = elo.get_rankings().head(top_n)
    
    print("\n" + "="*60)
    print(f"CURRENT TOP {top_n} TEAMS")
    print("="*60)
    print(rankings.to_string(index=False))
    print()


def predict_multiple_games(elo, matchups):
    """
    Predict multiple games.
    
    Args:
        elo: EloRatingSystem
        matchups: List of tuples (away_team, home_team)
    """
    print("\n" + "="*60)
    print("MULTIPLE GAME PREDICTIONS")
    print("="*60)
    
    for away, home in matchups:
        pred = predict_game(elo, home, away)
        fav = pred['favorite']
        prob = pred['home_win_probability'] if fav == home else pred['away_win_probability']
        spread = pred['spread_estimate']
        
        print(f"{away:>4} @ {home:<4} | {fav} favored {prob:.1%} | Spread: {home} {spread:+.1f}")
    
    print("="*60)


def interactive_mode(elo):
    """Interactive prediction mode."""
    print("\n" + "="*60)
    print("INTERACTIVE PREDICTION MODE")
    print("="*60)
    print("Enter matchups as: AWAY @ HOME (e.g., BAL @ MIA)")
    print("Type 'quit' to exit")
    print("="*60)
    
    while True:
        try:
            user_input = input("\nEnter matchup: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            # Parse input
            if '@' not in user_input:
                print("Invalid format. Use: AWAY @ HOME")
                continue
            
            parts = user_input.split('@')
            away = parts[0].strip().upper()
            home = parts[1].strip().upper()
            
            # Validate teams
            if away not in elo.ratings or home not in elo.ratings:
                print(f"Invalid team code. Available teams: {', '.join(sorted(elo.ratings.keys()))}")
                continue
            
            # Make prediction
            pred = predict_game(elo, home, away)
            format_prediction(pred, home, away)
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict NFL game outcomes')
    parser.add_argument('--season', type=int, default=2025, help='Season to analyze')
    parser.add_argument('--k', type=int, default=20, help='K-factor')
    parser.add_argument('--home-adv', type=int, default=40, help='Home advantage')
    parser.add_argument('--rankings', action='store_true', help='Show current rankings')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    parser.add_argument('--predict', nargs=2, metavar=('AWAY', 'HOME'), 
                       help='Predict single game (e.g., --predict BAL MIA)')
    
    args = parser.parse_args()
    
    # Calculate current Elo ratings
    elo = calculate_current_elo(args.season, args.k, args.home_adv)
    
    # Show rankings if requested
    if args.rankings:
        show_current_rankings(elo, top_n=32)
    
    # Single game prediction
    if args.predict:
        away_team, home_team = args.predict
        pred = predict_game(elo, home_team.upper(), away_team.upper())
        format_prediction(pred, home_team.upper(), away_team.upper())
    
    # Interactive mode
    if args.interactive:
        interactive_mode(elo)
    
    # Default: show top 10 and start interactive mode
    if not args.rankings and not args.predict and not args.interactive:
        show_current_rankings(elo, top_n=10)
        print("\nUse --predict AWAY HOME for a specific game")
        print("Or use --interactive for interactive mode")
        print("\nExample: python -m scripts.predict_game --predict BAL MIA")
