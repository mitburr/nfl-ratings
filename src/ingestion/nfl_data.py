"""Fetch and load NFL data into the database."""
import nflreadpy as nfl
import pandas as pd
import numpy as np
from src.database.db_manager import DatabaseManager


class NFLDataIngestion:
    """Handle fetching and loading NFL data."""
    
    def __init__(self):
        self.db = DatabaseManager()
    
    def fetch_teams(self):
        """Fetch and load team information."""
        # nflreadpy provides team info
        teams = nfl.load_teams().to_pandas()
        
        # Prepare for database - nflreadpy may have different column names
        # Adjust based on actual schema
        if 'team_abbr' in teams.columns:
            teams_df = teams[['team_abbr', 'team_name', 'team_conf', 'team_division']].copy()
        else:
            # Alternative column names
            teams_df = teams[['team_id', 'team_name', 'conference', 'division']].copy()
        
        teams_df.columns = ['team_id', 'team_name', 'conference', 'division']
        teams_df = teams_df.drop_duplicates(subset=['team_id'])
        
        # Use upsert logic - update if exists, insert if not
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        for _, row in teams_df.iterrows():
            cursor.execute("""
                INSERT INTO teams (team_id, team_name, conference, division)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (team_id) 
                DO UPDATE SET 
                    team_name = EXCLUDED.team_name,
                    conference = EXCLUDED.conference,
                    division = EXCLUDED.division
            """, (row['team_id'], row['team_name'], row['conference'], row['division']))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"Loaded {len(teams_df)} teams")
        return teams_df
    
    def fetch_schedule(self, years):
        """
        Fetch and load game schedules.
        
        Args:
            years: list of years to fetch
        """
        print(f"Fetching schedule data for years: {years}")
        schedules = nfl.load_schedules(years).to_pandas()  # Convert Polars to pandas
        
        # Filter to regular season and playoffs
        games = schedules[schedules['game_type'].isin(['REG', 'WC', 'DIV', 'CONF', 'SB'])].copy()
        
        # Prepare for database
        games_df = games[[
            'game_id', 'season', 'week', 'game_type', 'gameday',
            'home_team', 'away_team', 'home_score', 'away_score',
            'stadium', 'roof', 'surface', 'temp', 'wind'
        ]].copy()
        
        games_df.columns = [
            'game_id', 'season', 'week', 'game_type', 'game_date',
            'home_team', 'away_team', 'home_score', 'away_score',
            'stadium', 'roof', 'surface', 'temp', 'wind'
        ]
        
        # Use upsert logic - update if exists, insert if not
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        for _, row in games_df.iterrows():
            # Explicitly convert to Python native types to avoid PostgreSQL overflow
            values = (
                str(row['game_id']),
                int(row['season']),
                int(row['week']),
                str(row['game_type']) if pd.notna(row['game_type']) else None,
                row['game_date'],
                str(row['home_team']),
                str(row['away_team']),
                int(row['home_score']) if pd.notna(row['home_score']) else None,
                int(row['away_score']) if pd.notna(row['away_score']) else None,
                str(row['stadium']) if pd.notna(row['stadium']) else None,
                str(row['roof']) if pd.notna(row['roof']) else None,
                str(row['surface']) if pd.notna(row['surface']) else None,
                float(row['temp']) if pd.notna(row['temp']) else None,
                float(row['wind']) if pd.notna(row['wind']) else None
            )
            
            cursor.execute("""
                INSERT INTO games (
                    game_id, season, week, game_type, game_date,
                    home_team, away_team, home_score, away_score,
                    stadium, roof, surface, temp, wind
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (game_id) 
                DO UPDATE SET
                    home_score = EXCLUDED.home_score,
                    away_score = EXCLUDED.away_score,
                    updated_at = CURRENT_TIMESTAMP
            """, values)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"Loaded {len(games_df)} games")
        return games_df
    
    def fetch_play_by_play(self, years):
        """
        Fetch and load play-by-play data (includes EPA).
        
        Args:
            years: list of years to fetch
        """
        print(f"Fetching play-by-play data for years: {years}")
        print("Warning: This may take several minutes for multiple years...")
        
        pbp = nfl.load_pbp(years).to_pandas()  # Convert Polars to pandas
        
        # Select relevant columns
        plays_df = pbp[[
            'play_id', 'game_id', 'play_type', 'posteam', 'defteam',
            'quarter_seconds_remaining', 'down', 'ydstogo', 'yardline_100',
            'yards_gained', 'epa', 'wpa', 'success'
        ]].copy()
        
        # Rename for database
        plays_df.columns = [
            'play_id', 'game_id', 'play_type', 'posteam', 'defteam',
            'quarter', 'down', 'yards_to_go', 'yardline_100',
            'yards_gained', 'epa', 'wpa', 'success'
        ]
        
        # Remove plays with missing teams (kickoffs, etc.)
        plays_df = plays_df.dropna(subset=['posteam', 'defteam'])
        
        # Convert success to boolean
        plays_df['success'] = plays_df['success'].fillna(False).astype(bool)
        
        # Use bulk insert with ON CONFLICT DO NOTHING (plays shouldn't change)
        # This is more efficient than row-by-row for large datasets
        self.db.insert_dataframe(plays_df, 'plays', if_exists='append')
        
        print(f"Loaded {len(plays_df)} plays")
        return plays_df
    
    def load_current_season(self):
        """Load current season data (convenience method)."""
        current_year = 2025
        
        print("Loading teams...")
        self.fetch_teams()
        
        print("\nLoading games...")
        self.fetch_schedule([current_year])
        
        print("\nComplete! Ready for analysis.")


if __name__ == "__main__":
    ingestion = NFLDataIngestion()
    
    # Load current season
    ingestion.load_current_season()
    
    # Optional: Uncomment to load play-by-play (takes longer)
    # ingestion.fetch_play_by_play([2025])