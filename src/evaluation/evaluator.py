"""Model evaluation framework with proper cross-validation."""

import logging
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from src.models.base import BasePredictor
from src.database.db_manager import DatabaseManager
from src.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate prediction models with rolling cross-validation."""
    
    def __init__(self):
        self.db = DatabaseManager()
    
    def _load_season_games(self, season: int) -> pd.DataFrame:
        """Load all completed games for a season."""
        query = """
            SELECT game_id, season, week, game_date,
                   home_team, away_team, home_score, away_score
            FROM games
            WHERE season = %s AND home_score IS NOT NULL
            ORDER BY week, game_date
        """
        return self.db.query_to_dataframe(query, params=(season,))
    
    def _calculate_log_loss(self, predicted_prob: float, actual: bool) -> float:
        """Calculate log loss for single prediction."""
        p = predicted_prob if actual else 1 - predicted_prob
        return -np.log(max(p, 1e-15))
    
    def evaluate_season(
        self,
        model: BasePredictor,
        season: int,
        rolling: bool = True,
        metrics: Optional[List[str]] = None
    ) -> Dict:
        """Evaluate model on season with rolling or static fit.
        
        Args:
            model: Predictor instance
            season: Season to evaluate
            rolling: If True, refit model each week (prevents leakage)
            metrics: List of metric names to compute (default: all)
            
        Returns:
            Dictionary with evaluation results
        """
        if metrics is None:
            metrics = ['accuracy', 'log_loss', 'brier_score']
        
        logger.info(f"Evaluating {model.name} on {season} (rolling={rolling})")
        
        games = self._load_season_games(season)
        
        if len(games) == 0:
            logger.warning(f"No games found for season {season}")
            return None
        
        results = []
        current_week = None
        
        for _, game in games.iterrows():
            # Refit if week changed (rolling evaluation)
            if rolling and game['week'] != current_week:
                current_week = game['week']
                model.fit(season, through_week=current_week)
                logger.debug(f"Refitted model through week {current_week}")
            elif not rolling and not model.is_fitted:
                # Static evaluation - fit once on all prior data
                model.fit(season)
            
            # Make prediction
            pred = model.predict(game['home_team'], game['away_team'])
            
            # Record result
            actual_home_win = game['home_score'] > game['away_score']
            predicted_home_win = pred['home_win_prob'] > 0.5
            
            results.append({
                'week': game['week'],
                'game_id': game['game_id'],
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'predicted_prob': pred['home_win_prob'],
                'actual_result': 1 if actual_home_win else 0,
                'correct': predicted_home_win == actual_home_win,
                'log_loss': self._calculate_log_loss(pred['home_win_prob'], actual_home_win),
                **pred.get('metadata', {})
            })
        
        df = pd.DataFrame(results)
        
        # Compute metrics
        metric_values = compute_metrics(df, metrics)
        
        return {
            'model': model.name,
            'season': season,
            'rolling': rolling,
            **metric_values,
            'total_games': len(df),
            'by_week': df.groupby('week').agg({
                'correct': 'mean',
                'log_loss': 'mean'
            }),
            'predictions': df
        }
    
    def compare_models(
        self,
        models: List[BasePredictor],
        seasons: List[int],
        rolling: bool = True,
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Compare multiple models across seasons.
        
        Args:
            models: List of predictor instances
            seasons: List of seasons to evaluate
            rolling: Use rolling evaluation
            metrics: List of metrics to compute
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for season in seasons:
            for model in models:
                logger.info(f"Evaluating {model.name} on {season}")
                
                eval_result = self.evaluate_season(
                    model, season, rolling=rolling, metrics=metrics
                )
                
                if eval_result is not None:
                    results.append({
                        'model': model.name,
                        'season': season,
                        **{k: v for k, v in eval_result.items() 
                           if k not in ['model', 'season', 'by_week', 'predictions']},
                        'config': str(model.config)
                    })
        
        return pd.DataFrame(results)