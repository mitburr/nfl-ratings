"""Elo rating system for NFL teams."""
import pandas as pd
import numpy as np

# Fix for numpy 2.x compatibility with psycopg2
np.set_printoptions(legacy="1.25")

from src.database.db_manager import DatabaseManager


class EloRatingSystem:
    """Calculate and track Elo ratings for NFL teams."""
    
    def __init__(self, k_factor=20, home_advantage=65, initial_rating=1500, 
                 mov_multiplier=True):
        """
        Initialize Elo rating system.
        
        Args:
            k_factor: How much ratings change per game (default 20)
            home_advantage: Points added to home team's rating (default 65)
            initial_rating: Starting rating for all teams (default 1500)
            mov_multiplier: Whether to scale by margin of victory (default True)
        """
        self.k = k_factor
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating
        self.mov_multiplier = mov_multiplier
        self.db = DatabaseManager()
        
        # Current ratings (team_id -> rating)
        self.ratings = {}
        
        # History for tracking (list of dicts)
        self.history = []
    
    def get_rating(self, team):
        """Get current rating for a team, initialize if new."""
        if team not in self.ratings:
            self.ratings[team] = self.initial_rating
        return self.ratings[team]
    
    def expected_score(self, rating_a, rating_b):
        """
        Calculate expected win probability for team A vs team B.
        
        Uses standard Elo formula: E = 1 / (1 + 10^((Rb - Ra) / 400))
        """
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def margin_of_victory_multiplier(self, point_diff, winning_rating, losing_rating):
        """
        Calculate margin of victory multiplier.
        
        Larger wins matter more, but diminishing returns.
        Also scales based on expected outcome - bigger upsets = bigger changes.
        """
        if not self.mov_multiplier:
            return 1.0
        
        # Base multiplier from point differential (log scale for diminishing returns)
        mov = np.log(abs(point_diff) + 1)
        
        # Calculate rating difference (absolute value - we just care about magnitude)
        rating_diff = abs(winning_rating - losing_rating)
        
        # Scale by expected outcome - closer games (small rating diff) get higher multiplier
        # This formula gives ~1.0 for evenly matched teams, up to ~2.5 for big upsets
        multiplier = mov * (2.2 / ((rating_diff / 25.0) + 2.2))
        
        return max(1.0, multiplier)
    
    def update_ratings(self, home_team, away_team, home_score, away_score, 
                       week, season, game_id=None, debug_team=None):
        """
        Update ratings after a game.
        
        Args:
            debug_team: Team to log detailed info for (e.g., 'MIA')
        
        Returns: dict with game analysis
        """
        # Get current ratings
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)
        
        # Debug logging
        if debug_team and (home_team == debug_team or away_team == debug_team):
            print(f"\n{'='*60}")
            print(f"Week {week}: {away_team} @ {home_team} ({away_score}-{home_score})")
            print(f"Before: {home_team}={home_rating:.1f}, {away_team}={away_rating:.1f}")
        
        # Calculate expected outcome (with home advantage)
        expected_home = self.expected_score(
            home_rating + self.home_advantage,
            away_rating
        )
        expected_away = 1 - expected_home
        
        # Actual result (1 = win, 0 = loss, 0.5 = tie)
        if home_score > away_score:
            actual_home = 1.0
            actual_away = 0.0
            winner = home_team
            loser = away_team
            winner_rating = home_rating
            loser_rating = away_rating
        elif away_score > home_score:
            actual_home = 0.0
            actual_away = 1.0
            winner = away_team
            loser = home_team
            winner_rating = away_rating
            loser_rating = home_rating
        else:
            actual_home = 0.5
            actual_away = 0.5
            winner = None
            loser = None
            winner_rating = home_rating
            loser_rating = away_rating
        
        # Calculate margin of victory multiplier
        point_diff = abs(home_score - away_score)
        mov_mult = self.margin_of_victory_multiplier(
            point_diff, winner_rating, loser_rating
        ) if winner else 1.0
        
        # Calculate rating changes
        home_change = self.k * mov_mult * (actual_home - expected_home)
        away_change = self.k * mov_mult * (actual_away - expected_away)
        
        # Update ratings
        new_home_rating = home_rating + home_change
        new_away_rating = away_rating + away_change
        
        self.ratings[home_team] = new_home_rating
        self.ratings[away_team] = new_away_rating
        
        # Debug logging
        if debug_team and (home_team == debug_team or away_team == debug_team):
            print(f"Expected: {home_team}={expected_home:.1%}, {away_team}={expected_away:.1%}")
            print(f"Actual: {home_team} {'WON' if home_score > away_score else 'LOST'} {home_score}-{away_score}")
            print(f"MOV Multiplier: {mov_mult:.2f}")
            print(f"Changes: {home_team}={home_change:+.1f}, {away_team}={away_change:+.1f}")
            print(f"After: {home_team}={new_home_rating:.1f}, {away_team}={new_away_rating:.1f}")
        
        # Record in history
        game_info = {
            'game_id': game_id,
            'season': season,
            'week': week,
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score,
            'home_rating_before': home_rating,
            'away_rating_before': away_rating,
            'home_rating_after': new_home_rating,
            'away_rating_after': new_away_rating,
            'home_change': home_change,
            'away_change': away_change,
            'expected_home_win_prob': expected_home,
            'point_diff': point_diff,
            'mov_multiplier': mov_mult
        }
        self.history.append(game_info)
        
        return game_info
    
    def calculate_season(self, season, through_week=None, save_to_db=True, debug_team=None):
        """
        Calculate Elo ratings for a season, optionally through a specific week.
        """
        print(f"Calculating Elo ratings for {season} season...")
        if through_week:
            print(f"Limiting to games through Week {through_week}")

        query = """
            SELECT game_id, season, week, game_date,
                home_team, away_team, home_score, away_score
            FROM games
            WHERE season = %s AND home_score IS NOT NULL
            {}
            ORDER BY week, game_date
        """.format("AND week <= %s" if through_week else "")

        params = (season, through_week) if through_week else (season,)
        games = self.db.query_to_dataframe(query, params=params)
        
        if len(games) == 0:
            print(f"No completed games found for {season} (through_week={through_week})")
            return

        for _, game in games.iterrows():
            self.update_ratings(
                game['home_team'],
                game['away_team'],
                game['home_score'],
                game['away_score'],
                game['week'],
                game['season'],
                game['game_id'],
                debug_team=debug_team
            )
        
        if save_to_db:
            self.save_ratings_to_db(season)

        print(f"✓ Processed {len(games)} games (through_week={through_week})")
        print(f"✓ Rated {len(self.ratings)} teams")

    
    def save_ratings_to_db(self, season):
        """Save current ratings to database."""
        # Get the latest week from history
        if not self.history:
            return
        
        latest_week = max(h['week'] for h in self.history)
        
        # Prepare data for insertion
        ratings_data = []
        for team_id, rating in self.ratings.items():
            games_played = sum(
                1 for h in self.history 
                if h['home_team'] == team_id or h['away_team'] == team_id
            )
            # Convert numpy types to Python native types
            ratings_data.append((
                team_id, 
                int(season), 
                int(latest_week), 
                float(rating),  # Convert numpy float to Python float
                int(games_played)
            ))
        
        # Insert using upsert
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        for data in ratings_data:
            cursor.execute("""
                INSERT INTO elo_ratings (team_id, season, week, rating, games_played)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (team_id, season, week)
                DO UPDATE SET
                    rating = EXCLUDED.rating,
                    games_played = EXCLUDED.games_played
            """, data)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"✓ Saved ratings to database (season {season}, week {latest_week})")
    
    def get_rankings(self):
        """Get current rankings as a DataFrame."""
        rankings = pd.DataFrame([
            {
                'team': team,
                'rating': rating,
                'rank': 0  # Will be set after sorting
            }
            for team, rating in self.ratings.items()
        ])
        
        rankings = rankings.sort_values('rating', ascending=False).reset_index(drop=True)
        rankings['rank'] = range(1, len(rankings) + 1)
        
        return rankings[['rank', 'team', 'rating']]
    
    def predict_game(self, home_team, away_team):
        """
        Predict the outcome of a game.
        
        Returns: dict with prediction details
        """
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)
        
        home_win_prob = self.expected_score(
            home_rating + self.home_advantage,
            away_rating
        )
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_rating': home_rating,
            'away_rating': away_rating,
            'home_win_probability': home_win_prob,
            'away_win_probability': 1 - home_win_prob,
            'favorite': home_team if home_win_prob > 0.5 else away_team,
            'spread_estimate': (home_rating - away_rating + self.home_advantage) / 25
        }
    
    def get_biggest_upsets(self, n=10):
        """Get the N biggest upsets based on pre-game expected win probability."""
        if not self.history:
            return pd.DataFrame()
        
        upsets = []
        for game in self.history:
            # Skip ties
            if game['home_score'] == game['away_score']:
                continue
            
            # Determine if upset occurred
            home_won = game['home_score'] > game['away_score']
            expected_home_win = game['expected_home_win_prob']
            
            if home_won and expected_home_win < 0.5:
                upset_factor = 0.5 - expected_home_win
                upsets.append({
                    'week': game['week'],
                    'winner': game['home_team'],
                    'loser': game['away_team'],
                    'score': f"{game['home_score']}-{game['away_score']}",
                    'upset_factor': upset_factor,
                    'win_probability': expected_home_win
                })
            elif not home_won and expected_home_win > 0.5:
                upset_factor = expected_home_win - 0.5
                upsets.append({
                    'week': game['week'],
                    'winner': game['away_team'],
                    'loser': game['home_team'],
                    'score': f"{game['away_score']}-{game['home_score']}",
                    'upset_factor': upset_factor,
                    'win_probability': 1 - expected_home_win
                })
        
        df = pd.DataFrame(upsets)
        return df.nlargest(n, 'upset_factor') if len(df) > 0 else df


if __name__ == "__main__":
    # Example usage
    elo = EloRatingSystem(k_factor=20, home_advantage=65)
    
    # Calculate ratings for 2024 season with DEBUG for Miami
    elo.calculate_season(2024, save_to_db=True, debug_team='None')
    
    # Show current rankings
    print("\n" + "="*50)
    print("2024 NFL ELO RANKINGS")
    print("="*50)
    rankings = elo.get_rankings()
    print(rankings.to_string(index=False))
    
    # Show biggest upsets
    print("\n" + "="*50)
    print("BIGGEST UPSETS OF 2024")
    print("="*50)
    upsets = elo.get_biggest_upsets(5)
    if len(upsets) > 0:
        print(upsets.to_string(index=False))
    
    # Example prediction
    print("\n" + "="*50)
    print("EXAMPLE PREDICTION: Chiefs vs Bills")
    print("="*50)
    prediction = elo.predict_game('KC', 'BUF')
    print(f"Home: {prediction['home_team']} ({prediction['home_rating']:.1f})")
    print(f"Away: {prediction['away_team']} ({prediction['away_rating']:.1f})")
    print(f"Home win probability: {prediction['home_win_probability']:.1%}")
    print(f"Estimated spread: {prediction['spread_estimate']:.1f}")
