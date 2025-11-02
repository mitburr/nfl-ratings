"""Elo rating predictor for NFL games."""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.models.base import BasePredictor
from src.database.db_manager import DatabaseManager
from src.utils.cache import EloCache

logger = logging.getLogger(__name__)


class EloPredictor(BasePredictor):
    """Elo-based prediction system with caching support."""
    
    def __init__(self, config: Dict):
        """Initialize Elo predictor.
        
        Args:
            config: Dictionary with keys:
                - k_factor: Rating change multiplier (default: 20)
                - home_advantage: Points for home team (default: 40)
                - initial_rating: Starting rating (default: 1500)
                - mov_multiplier: Scale by margin of victory (default: True)
        """
        super().__init__(config)
        self.k_factor = config.get('k_factor', 20)
        self.home_advantage = config.get('home_advantage', 40)
        self.initial_rating = config.get('initial_rating', 1500)
        self.mov_multiplier = config.get('mov_multiplier', True)
        
        self.ratings = {}
        self.history = []
        self.db = DatabaseManager()
        self.cache = EloCache()
    
    def get_rating(self, team: str) -> float:
        """Get current rating for team, initialize if new."""
        if team not in self.ratings:
            self.ratings[team] = self.initial_rating
        return self.ratings[team]
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected win probability for team A vs B."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def margin_of_victory_multiplier(
        self, 
        point_diff: int, 
        winning_rating: float, 
        losing_rating: float
    ) -> float:
        """Calculate MOV multiplier for rating changes."""
        if not self.mov_multiplier:
            return 1.0
        
        mov = np.log(abs(point_diff) + 1)
        rating_diff = abs(winning_rating - losing_rating)
        multiplier = mov * (2.2 / ((rating_diff / 25.0) + 2.2))
        
        return max(1.0, multiplier)
    
    def update_ratings(
        self,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        week: int,
        season: int,
        game_id: Optional[str] = None
    ) -> Dict:
        """Update ratings after game completion."""
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)
        
        # Expected outcome
        expected_home = self.expected_score(
            home_rating + self.home_advantage,
            away_rating
        )
        
        # Actual result
        if home_score > away_score:
            actual_home = 1.0
            winner_rating = home_rating
            loser_rating = away_rating
        elif away_score > home_score:
            actual_home = 0.0
            winner_rating = away_rating
            loser_rating = home_rating
        else:
            actual_home = 0.5
            winner_rating = home_rating
            loser_rating = away_rating
        
        # MOV multiplier
        point_diff = abs(home_score - away_score)
        mov_mult = self.margin_of_victory_multiplier(
            point_diff, winner_rating, loser_rating
        ) if actual_home != 0.5 else 1.0
        
        # Rating changes
        home_change = self.k_factor * mov_mult * (actual_home - expected_home)
        away_change = self.k_factor * mov_mult * ((1 - actual_home) - (1 - expected_home))
        
        # Update
        self.ratings[home_team] = home_rating + home_change
        self.ratings[away_team] = away_rating + away_change
        
        # Record
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
            'home_rating_after': self.ratings[home_team],
            'away_rating_after': self.ratings[away_team],
            'expected_home_win_prob': expected_home
        }
        self.history.append(game_info)
        
        return game_info
    
    def fit(self, season: int, through_week: Optional[int] = None):
        """Calculate Elo ratings through specified week."""
        # Check cache first
        cached = self.cache.get(season, through_week, self.config)
        if cached is not None:
            logger.info(f"Loaded Elo cache for {season} week {through_week}")
            self.ratings = cached['ratings']
            self.history = cached['history']
            self.is_fitted = True
            return self
        
        logger.info(f"Calculating Elo for {season} through week {through_week}")
        
        # Load games
        week_filter = "AND week <= %s" if through_week else ""
        params = (season, through_week) if through_week else (season,)
        
        query = f"""
            SELECT game_id, season, week, game_date,
                   home_team, away_team, home_score, away_score
            FROM games
            WHERE season = %s AND home_score IS NOT NULL {week_filter}
            ORDER BY week, game_date
        """
        
        games = self.db.query_to_dataframe(query, params=params)
        
        if len(games) == 0:
            logger.warning(f"No completed games found for {season} through week {through_week}")
            return self
        
        # Process games
        for _, game in games.iterrows():
            self.update_ratings(
                game['home_team'],
                game['away_team'],
                game['home_score'],
                game['away_score'],
                game['week'],
                game['season'],
                game['game_id']
            )
        
        # Cache results
        self.cache.save(season, through_week, self.config, {
            'ratings': self.ratings.copy(),
            'history': self.history.copy()
        })
        
        self.is_fitted = True
        logger.info(f"Processed {len(games)} games, rated {len(self.ratings)} teams")
        
        return self
    
    def predict(self, home_team: str, away_team: str) -> Dict:
        """Predict game outcome using current Elo ratings."""
        if not self.is_fitted:
            logger.warning("Model not fitted, using default ratings")
        
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)
        
        home_win_prob = self.expected_score(
            home_rating + self.home_advantage,
            away_rating
        )
        
        spread = (home_rating - away_rating + self.home_advantage) / 25
        confidence = abs(home_win_prob - 0.5) * 2
        
        return {
            'home_win_prob': home_win_prob,
            'away_win_prob': 1 - home_win_prob,
            'spread': spread,
            'confidence': confidence,
            'metadata': {
                'home_elo': home_rating,
                'away_elo': away_rating,
                'home_advantage': self.home_advantage
            }
        }
    
    def get_rankings(self) -> pd.DataFrame:
        """Get current team rankings."""
        rankings = pd.DataFrame([
            {'team': team, 'rating': rating}
            for team, rating in self.ratings.items()
        ])
        rankings = rankings.sort_values('rating', ascending=False).reset_index(drop=True)
        rankings['rank'] = range(1, len(rankings) + 1)
        return rankings[['rank', 'team', 'rating']]
    
    def get_state(self) -> Dict:
        """Get model state for serialization."""
        return {
            'ratings': self.ratings,
            'history': self.history,
            'config': self.config
        }
    
    def load_state(self, state: Dict):
        """Restore model from saved state."""
        self.ratings = state['ratings']
        self.history = state['history']
        self.is_fitted = True