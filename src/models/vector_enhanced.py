"""Vector-enhanced Elo predictor."""

import logging
from typing import Dict, Optional

import pandas as pd

from src.models.base import BasePredictor
from src.models.elo import EloPredictor
from src.analysis.team_vectors import calculate_team_vectors, analyze_matchup

logger = logging.getLogger(__name__)


class VectorEnhancedPredictor(BasePredictor):
    """Elo predictor with vector-based adjustments."""
    
    def __init__(self, config: Dict):
        """Initialize vector-enhanced predictor.
        
        Args:
            config: Dictionary with keys:
                - elo_config: Config dict for EloPredictor
                - vector_config: Dict with:
                    - boost: Multiplier for vector advantage (default: 0.60)
                    - weight: Fraction of boost to apply (default: 0.35)
                    - blend_prior: Use weighted blend with prior season (default: False)
                    - prior_weight: Weight for prior season if blending (default: 0.25)
        """
        super().__init__(config)
        self.elo = EloPredictor(config.get('elo_config', {}))
        
        vector_config = config.get('vector_config', {})
        self.boost = vector_config.get('boost', 0.60)
        self.weight = vector_config.get('weight', 0.35)
        self.blend_prior = vector_config.get('blend_prior', False)
        self.prior_weight = vector_config.get('prior_weight', 0.25)
        
        self.vectors = None
        self.current_season = None
        self.current_week = None
    
    def _calculate_weighted_vectors(
        self,
        prior_season: int,
        current_season: int,
        through_week: int
    ) -> pd.DataFrame:
        """Blend prior season and current season vectors."""
        pd.set_option('future.no_silent_downcasting', True)
        
        prior = calculate_team_vectors(prior_season)
        current = calculate_team_vectors(current_season, through_week=through_week)
        
        merged = prior.merge(
            current,
            on='team',
            suffixes=('_prior', '_current'),
            how='outer'
        )
        
        # Fill missing values
        for suffix in ['_prior', '_current']:
            for col in merged.columns:
                if col.endswith(suffix):
                    merged[col] = merged[col].fillna(merged[col].mean())
        
        merged = merged.infer_objects(copy=False)
        
        # Blend metrics
        metrics = [
            'rush_ypa', 'pass_ypa', 'rush_ypa_allowed', 'pass_ypa_allowed',
            'success_rate', 'explosive_rate', 'success_rate_allowed', 'explosive_rate_allowed'
        ]
        
        for metric in metrics:
            prior_col = f'{metric}_prior'
            current_col = f'{metric}_current'
            if prior_col in merged.columns and current_col in merged.columns:
                merged[metric] = (
                    merged[prior_col] * self.prior_weight +
                    merged[current_col] * (1 - self.prior_weight)
                )
        
        final_cols = ['team'] + metrics
        return merged[[col for col in final_cols if col in merged.columns]]
    
    def fit(self, season: int, through_week: Optional[int] = None):
        """Fit both Elo and calculate vectors."""
        logger.info(f"Fitting VectorEnhanced for {season} through week {through_week}")
        
        # Fit Elo
        self.elo.fit(season, through_week)
        
        # Calculate vectors
        if self.blend_prior:
            self.vectors = self._calculate_weighted_vectors(
                season - 1, season, through_week
            )
            logger.info("Using weighted vector blend (prior + current)")
        else:
            self.vectors = calculate_team_vectors(season, through_week=through_week)
            logger.info("Using cumulative vectors (current season only)")
        
        self.current_season = season
        self.current_week = through_week
        self.is_fitted = True
        
        return self
    
    def predict(self, home_team: str, away_team: str) -> Dict:
        """Predict with Elo + vector adjustment."""
        if not self.is_fitted:
            logger.warning("Model not fitted, returning Elo-only prediction")
            return self.elo.predict(home_team, away_team)
        
        # Base Elo prediction
        elo_pred = self.elo.predict(home_team, away_team)
        
        # Apply vector adjustment
        matchup = analyze_matchup(home_team, away_team, self.vectors)
        
        if matchup is not None:
            vector_boost = matchup['total_advantage'] * self.boost
            adjustment = vector_boost * self.weight
            adjusted_prob = elo_pred['home_win_prob'] + adjustment
            adjusted_prob = max(0.01, min(0.99, adjusted_prob))
        else:
            adjusted_prob = elo_pred['home_win_prob']
            adjustment = 0.0
            logger.debug(f"No vector data for {home_team} vs {away_team}")
        
        spread = (adjusted_prob - 0.5) * 28
        confidence = abs(adjusted_prob - 0.5) * 2
        
        return {
            'home_win_prob': adjusted_prob,
            'away_win_prob': 1 - adjusted_prob,
            'spread': spread,
            'confidence': confidence,
            'metadata': {
                'elo_prob': elo_pred['home_win_prob'],
                'vector_adjustment': adjustment,
                'matchup_advantage': matchup['total_advantage'] if matchup else None,
                **elo_pred['metadata']
            }
        }