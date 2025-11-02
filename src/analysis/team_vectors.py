"""Calculate team statistical profiles (vectors) for matchup analysis."""
import pandas as pd
import numpy as np
from src.database.db_manager import DatabaseManager


def calculate_team_vectors(season=2024, through_week=None):
    """
    Calculate offensive/defensive efficiency vectors for each team.
    
    Args:
        season: Season year
        through_week: Only include games up to this week (None = all games)
    """
    db = DatabaseManager()
    
    week_filter = "AND g.week < %s" if through_week else ""
    params = (season, through_week) if through_week else (season,)
    
    # Get offensive stats
    offensive_query = f"""
        SELECT 
            p.posteam as team,
            AVG(CASE WHEN p.play_type = 'run' THEN p.yards_gained END)::NUMERIC as rush_ypa,
            AVG(CASE WHEN p.play_type = 'pass' THEN p.yards_gained END)::NUMERIC as pass_ypa,
            AVG(CASE WHEN p.success = true THEN 1.0 ELSE 0.0 END)::NUMERIC as success_rate,
            (SUM(CASE WHEN p.yards_gained >= 20 THEN 1 ELSE 0 END)::NUMERIC / COUNT(*)::NUMERIC) as explosive_rate,
            COUNT(*) as total_plays
        FROM plays p
        JOIN games g ON p.game_id = g.game_id
        WHERE g.season = %s {week_filter} AND p.posteam IS NOT NULL
        GROUP BY p.posteam
    """
    
    offense = db.query_to_dataframe(offensive_query, params=params)
    
    # Get defensive stats (what teams allow)
    defensive_query = f"""
        SELECT 
            p.defteam as team,
            AVG(CASE WHEN p.play_type = 'run' THEN p.yards_gained END)::NUMERIC as rush_ypa_allowed,
            AVG(CASE WHEN p.play_type = 'pass' THEN p.yards_gained END)::NUMERIC as pass_ypa_allowed,
            AVG(CASE WHEN p.success = true THEN 1.0 ELSE 0.0 END)::NUMERIC as success_rate_allowed,
            (SUM(CASE WHEN p.yards_gained >= 20 THEN 1 ELSE 0 END)::NUMERIC / COUNT(*)::NUMERIC) as explosive_rate_allowed
        FROM plays p
        JOIN games g ON p.game_id = g.game_id
        WHERE g.season = %s {week_filter} AND p.defteam IS NOT NULL
        GROUP BY p.defteam
    """
    
    defense = db.query_to_dataframe(defensive_query, params=params)
    
    # Merge offense and defense
    vectors = offense.merge(defense, on='team', how='outer')
    
    return vectors


def get_team_vector(team, vectors_df):
    """Get vector profile for a specific team."""
    team_data = vectors_df[vectors_df['team'] == team]
    if len(team_data) == 0:
        return None
    return team_data.iloc[0].to_dict()


def analyze_matchup(home_team, away_team, vectors_df):
    """
    Analyze matchup advantages between two teams.
    
    Returns dict with:
    - run_advantage: How well home runs vs away run defense
    - pass_advantage: How well home passes vs away pass defense  
    - total_advantage: Combined offensive advantage
    """
    home = get_team_vector(home_team, vectors_df)
    away = get_team_vector(away_team, vectors_df)
    
    if home is None or away is None:
        return None
    
    # Home team's offense vs away team's defense (absolute values)
    run_advantage = home['rush_ypa'] - away['rush_ypa_allowed']
    pass_advantage = home['pass_ypa'] - away['pass_ypa_allowed']
    
    # Away team's offense vs home team's defense (flip sign for home perspective)
    run_disadvantage = away['rush_ypa'] - home['rush_ypa_allowed']
    pass_disadvantage = away['pass_ypa'] - home['pass_ypa_allowed']
    
    # Net advantages (positive = home advantage)
    total_advantage = (run_advantage + pass_advantage - run_disadvantage - pass_disadvantage) / 4
    
    return {
        'home_run_advantage': run_advantage,
        'home_pass_advantage': pass_advantage,
        'away_run_advantage': run_disadvantage,
        'away_pass_advantage': pass_disadvantage,
        'total_advantage': total_advantage
    }


def display_team_vectors(season=2024, top_n=10):
    """Display top teams by offensive/defensive efficiency."""
    vectors = calculate_team_vectors(season)
    
    print("\n" + "="*70)
    print(f"TOP {top_n} RUSHING OFFENSES ({season})")
    print("="*70)
    top_rush = vectors.nlargest(top_n, 'rush_ypa')[['team', 'rush_ypa']]
    print(top_rush.to_string(index=False))
    
    print("\n" + "="*70)
    print(f"TOP {top_n} PASSING OFFENSES ({season})")
    print("="*70)
    top_pass = vectors.nlargest(top_n, 'pass_ypa')[['team', 'pass_ypa']]
    print(top_pass.to_string(index=False))
    
    print("\n" + "="*70)
    print(f"TOP {top_n} RUSHING DEFENSES ({season})")
    print("="*70)
    top_rush_def = vectors.nsmallest(top_n, 'rush_ypa_allowed')[['team', 'rush_ypa_allowed']]
    print(top_rush_def.to_string(index=False))
    
    print("\n" + "="*70)
    print(f"TOP {top_n} PASSING DEFENSES ({season})")
    print("="*70)
    top_pass_def = vectors.nsmallest(top_n, 'pass_ypa_allowed')[['team', 'pass_ypa_allowed']]
    print(top_pass_def.to_string(index=False))
    
    return vectors


if __name__ == "__main__":
    # Example usage
    vectors = display_team_vectors(2024)
    
    # Example matchup analysis
    print("\n" + "="*70)
    print("EXAMPLE MATCHUP: KC vs BUF")
    print("="*70)
    matchup = analyze_matchup('KC', 'BUF', vectors)
    if matchup:
        print(f"KC run advantage: {matchup['home_run_advantage']:+.3f}")
        print(f"KC pass advantage: {matchup['home_pass_advantage']:+.3f}")
        print(f"Total advantage: {matchup['total_advantage']:+.3f} (positive = KC favored)")