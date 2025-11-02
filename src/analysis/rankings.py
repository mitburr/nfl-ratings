"""Analyze and visualize Elo ratings."""
import pandas as pd
from src.models.elo import EloRatingSystem
from src.database.db_manager import DatabaseManager


def compare_elo_to_records(season=2024, through_week=None):
    """Compare Elo rankings (up to given week) to actual win-loss records."""
    db = DatabaseManager()
    elo = EloRatingSystem()
    
    # Calculate Elo up to given week
    elo.calculate_season(season, through_week=through_week, save_to_db=False)
    elo_rankings = elo.get_rankings()
    
    # Get records up to the same week
    week_filter = "AND g.week <= %s" if through_week else ""
    params = (season, through_week, season, through_week) if through_week else (season, season)
    
    query = f"""
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
            FROM games g
            WHERE g.season = %s {week_filter} AND g.home_score IS NOT NULL
            
            UNION ALL
            
            SELECT 
                away_team as team,
                CASE WHEN away_score > home_score THEN 1 ELSE 0 END as wins,
                CASE WHEN away_score < home_score THEN 1 ELSE 0 END as losses
            FROM games g
            WHERE g.season = %s {week_filter} AND g.home_score IS NOT NULL
        ) combined
        GROUP BY team
        ORDER BY win_pct DESC, wins DESC
    """
    
    records = db.query_to_dataframe(query, params=params)
    
    # Merge
    comparison = elo_rankings.merge(records, on='team', how='left')
    comparison['record_rank'] = comparison['win_pct'].rank(ascending=False, method='min')
    comparison['rank_diff'] = comparison['record_rank'] - comparison['rank']
    comparison['record'] = (
        comparison['wins'].astype(int).astype(str) + '-' + comparison['losses'].astype(int).astype(str)
    )
    
    return comparison[['rank', 'team', 'rating', 'record', 'win_pct', 'record_rank', 'rank_diff']].sort_values('rank')


def find_most_overrated_underrated(season=2024):
    """Find teams most over/under-rated by their record."""
    comparison = compare_elo_to_records(season)
    
    print("MOST UNDERRATED (Elo rank >> Record rank)")
    print("="*60)
    underrated = comparison.nsmallest(5, 'rank_diff')
    for _, row in underrated.iterrows():
        print(f"{row['team']:4} - Elo Rank: {int(row['rank']):2}, Record Rank: {int(row['record_rank']):2} ({row['record']})")
        print(f"       Better than record suggests by {abs(int(row['rank_diff']))} spots")
    
    print("\n" + "="*60)
    print("MOST OVERRATED (Elo rank << Record rank)")
    print("="*60)
    overrated = comparison.nlargest(5, 'rank_diff')
    for _, row in overrated.iterrows():
        print(f"{row['team']:4} - Elo Rank: {int(row['rank']):2}, Record Rank: {int(row['record_rank']):2} ({row['record']})")
        print(f"       Worse than record suggests by {int(row['rank_diff'])} spots")


def analyze_strength_of_schedule(season=2024):
    """Analyze which teams had the toughest schedules."""
    db = DatabaseManager()
    elo = EloRatingSystem()
    
    # Calculate Elo
    elo.calculate_season(season, save_to_db=False)
    
    # For each team, calculate average opponent rating
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
    
    sos_df = sos_df.sort_values('avg_opponent_rating', ascending=False)
    
    return sos_df


if __name__ == "__main__":
    print("NFL ELO RATING ANALYSIS")
    print("="*60)
    
    # Full comparison
    print("\nFULL COMPARISON: Elo vs Record")
    print("="*60)
    comparison = compare_elo_to_records(2024)
    print(comparison.to_string(index=False))
    
    # Over/under rated
    print("\n")
    find_most_overrated_underrated(2024)
    
    # Strength of schedule
    print("\n")
    print("STRENGTH OF SCHEDULE (by avg opponent Elo)")
    print("="*60)
    sos = analyze_strength_of_schedule(2024)
    print(sos.head(10).to_string(index=False))
    print("...")
    print(sos.tail(5).to_string(index=False))
