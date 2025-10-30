"""Run complete NFL Elo analysis pipeline."""
import sys
import pandas as pd
from src.models.elo import EloRatingSystem
from src.database.db_manager import DatabaseManager


def run_full_analysis(season=2024, k_factor=20, home_advantage=65, 
                     save_to_db=False, show_debug=None):
    """
    Run complete analysis pipeline.
    
    Args:
        season: Year to analyze
        k_factor: Elo k-factor (rating volatility)
        home_advantage: Home field advantage in Elo points
        save_to_db: Whether to save results to database
        show_debug: Team code to debug (e.g., 'MIA'), or None
    """
    print("="*70)
    print(f"NFL ELO ANALYSIS - {season} Season")
    print(f"Parameters: K={k_factor}, Home Advantage={home_advantage}")
    print("="*70)
    
    # Initialize Elo system
    elo = EloRatingSystem(k_factor=k_factor, home_advantage=home_advantage)
    
    # Calculate ratings
    print("\n[1/6] Calculating Elo ratings...")
    elo.calculate_season(season, save_to_db=save_to_db, debug_team=show_debug)
    
    # Check if any games were found
    if len(elo.ratings) == 0:
        print("\n" + "="*70)
        print("ERROR: No completed games found for this season")
        print("="*70)
        print(f"\nTry loading data first:")
        print(f"  python -m scripts.load_historical_data {season}")
        return None
    
    # Get rankings
    print("\n[2/6] Generating rankings...")
    rankings = elo.get_rankings()
    
    print("\n" + "="*70)
    print("ELO RANKINGS")
    print("="*70)
    print(rankings.to_string(index=False))
    
    # Get biggest upsets
    print("\n[3/6] Finding biggest upsets...")
    upsets = elo.get_biggest_upsets(10)
    
    print("\n" + "="*70)
    print("BIGGEST UPSETS")
    print("="*70)
    if len(upsets) > 0:
        print(upsets.to_string(index=False))
    else:
        print("No upsets found")
    
    # Compare to actual records
    print("\n[4/6] Comparing Elo to win-loss records...")
    comparison = compare_elo_to_records(elo, season)
    
    print("\n" + "="*70)
    print("ELO vs RECORD COMPARISON")
    print("="*70)
    print(comparison.to_string(index=False))
    
    # Most over/underrated
    print("\n[5/6] Finding over/underrated teams...")
    print("\n" + "="*70)
    print("MOST UNDERRATED (Better than record suggests)")
    print("="*70)
    underrated = comparison.nsmallest(5, 'rank_diff')
    for _, row in underrated.iterrows():
        diff = abs(int(row['rank_diff']))
        print(f"{row['team']:4} - Elo: #{int(row['rank']):2}  Record: #{int(row['record_rank']):2} ({row['record']})  → {diff} spots better")
    
    print("\n" + "="*70)
    print("MOST OVERRATED (Worse than record suggests)")
    print("="*70)
    overrated = comparison.nlargest(5, 'rank_diff')
    for _, row in overrated.iterrows():
        diff = int(row['rank_diff'])
        print(f"{row['team']:4} - Elo: #{int(row['rank']):2}  Record: #{int(row['record_rank']):2} ({row['record']})  → {diff} spots worse")
    
    # Strength of schedule
    print("\n[6/6] Calculating strength of schedule...")
    sos = analyze_strength_of_schedule(elo)
    
    print("\n" + "="*70)
    print("STRENGTH OF SCHEDULE (by avg opponent Elo)")
    print("="*70)
    print("Toughest schedules:")
    print(sos.head(5).to_string(index=False))
    print("\nEasiest schedules:")
    print(sos.tail(5).to_string(index=False))
    
    # Summary stats
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total teams rated: {len(rankings)}")
    print(f"Total games processed: {len(elo.history)}")
    print(f"Rating range: {rankings['rating'].min():.1f} - {rankings['rating'].max():.1f}")
    print(f"Average rating: {rankings['rating'].mean():.1f}")
    print(f"Rating std dev: {rankings['rating'].std():.1f}")
    
    return {
        'elo': elo,
        'rankings': rankings,
        'upsets': upsets,
        'comparison': comparison,
        'sos': sos
    }


def compare_elo_to_records(elo, season):
    """Compare Elo rankings to actual win-loss records."""
    db = DatabaseManager()
    
    # Get Elo rankings
    elo_rankings = elo.get_rankings()
    
    # Get actual records from database
    query = """
        SELECT 
            team,
            SUM(wins) as wins,
            SUM(losses) as losses,
            ROUND(CAST(SUM(wins) AS NUMERIC) / NULLIF(SUM(wins) + SUM(losses), 0), 3) as win_pct
        FROM (
            SELECT 
                home_team as team,
                CASE WHEN home_score > away_score THEN 1 ELSE 0 END as wins,
                CASE WHEN home_score < away_score THEN 1 ELSE 0 END as losses
            FROM games
            WHERE season = %s AND home_score IS NOT NULL
            
            UNION ALL
            
            SELECT 
                away_team as team,
                CASE WHEN away_score > home_score THEN 1 ELSE 0 END as wins,
                CASE WHEN away_score < home_score THEN 1 ELSE 0 END as losses
            FROM games
            WHERE season = %s AND home_score IS NOT NULL
        ) combined
        GROUP BY team
        ORDER BY win_pct DESC, wins DESC
    """
    
    records = db.query_to_dataframe(query, params=(season, season))
    
    # Merge Elo and records
    comparison = elo_rankings.merge(records, left_on='team', right_on='team', how='left')
    
    # Calculate rank differences
    comparison['record_rank'] = comparison['win_pct'].rank(ascending=False, method='min')
    comparison['rank_diff'] = comparison['record_rank'] - comparison['rank']
    
    # Format for display
    comparison['record'] = comparison['wins'].astype(int).astype(str) + '-' + comparison['losses'].astype(int).astype(str)
    
    return comparison[['rank', 'team', 'rating', 'record', 'win_pct', 'record_rank', 'rank_diff']].sort_values('rank')


def analyze_strength_of_schedule(elo):
    """Analyze which teams had the toughest schedules."""
    sos = {}
    
    for game in elo.history:
        home_team = game['home_team']
        away_team = game['away_team']
        
        # Home team faced away team
        if home_team not in sos:
            sos[home_team] = []
        sos[home_team].append(game['away_rating_before'])
        
        # Away team faced home team
        if away_team not in sos:
            sos[away_team] = []
        sos[away_team].append(game['home_rating_before'])
    
    # Calculate averages
    sos_df = pd.DataFrame([
        {
            'team': team,
            'avg_opponent_rating': sum(ratings) / len(ratings),
            'games': len(ratings)
        }
        for team, ratings in sos.items()
    ])
    
    return sos_df.sort_values('avg_opponent_rating', ascending=False)


if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Run NFL Elo analysis')
    parser.add_argument('--season', type=int, default=2024, help='Season to analyze')
    parser.add_argument('--k', type=int, default=20, help='K-factor (rating volatility)')
    parser.add_argument('--home', type=int, default=65, help='Home advantage in Elo points')
    parser.add_argument('--save', action='store_true', help='Save results to database')
    parser.add_argument('--debug', type=str, default=None, help='Team to debug (e.g., MIA)')
    
    args = parser.parse_args()
    
    # Run analysis
    results = run_full_analysis(
        season=args.season,
        k_factor=args.k,
        home_advantage=args.home,
        save_to_db=args.save,
        show_debug=args.debug
    )
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)
