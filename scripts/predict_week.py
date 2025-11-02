"""Predict upcoming NFL games using config-based predictor.

Usage:
    python -m scripts.predict_week --week 9 --season 2025 --config experiments/configs/elo_baseline.yaml
"""

import argparse
from datetime import datetime
import logging

import pandas as pd

from src.models.elo import EloPredictor
from src.models.vector_enhanced import VectorEnhancedPredictor
from src.database.db_manager import DatabaseManager
from src.utils.config import load_config, validate_config
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def create_model(config: dict):
    """Create model from config."""
    model_type = config['model_type']
    if model_type == 'elo':
        return EloPredictor(config['parameters'])
    elif model_type == 'vectors':
        return VectorEnhancedPredictor(config['parameters'])
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_upcoming_games(season: int, week: int) -> pd.DataFrame:
    """Get games for specified week."""
    db = DatabaseManager()
    query = """
        SELECT game_id, week, game_date, home_team, away_team
        FROM games
        WHERE season = %s AND week = %s
        ORDER BY game_date
    """
    return db.query_to_dataframe(query, params=(season, week))


def predict_week(config_path: str, season: int, week: int):
    """Generate predictions for a week."""
    # Load config
    config = validate_config(load_config(config_path))
    
    # Setup logging
    setup_logging('INFO')
    
    logger.info("="*70)
    logger.info(f"NFL WEEK {week} PREDICTIONS - {datetime.now().strftime('%B %d, %Y')}")
    logger.info(f"Using: {config['name']}")
    logger.info("="*70)
    
    # Create and fit model
    model = create_model(config)
    model.fit(season, through_week=week-1)
    
    logger.info(f"Model fitted through week {week-1}")
    
    # Get games
    games = get_upcoming_games(season, week)
    
    if len(games) == 0:
        logger.warning(f"No games found for week {week}")
        return
    
    # Make predictions
    predictions = []
    
    print("\n" + "="*70)
    print(f"PREDICTIONS FOR WEEK {week}")
    print("="*70)
    
    for _, game in games.iterrows():
        pred = model.predict(game['home_team'], game['away_team'])
        
        predictions.append({
            'date': game['game_date'],
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'home_prob': pred['home_win_prob'],
            'away_prob': pred['away_win_prob'],
            'spread': pred['spread'],
            'confidence': pred['confidence']
        })
        
        fav = game['home_team'] if pred['home_win_prob'] > 0.5 else game['away_team']
        fav_prob = max(pred['home_win_prob'], pred['away_win_prob'])
        
        print(f"\n{game['away_team']} @ {game['home_team']}")
        print(f"  Favorite: {fav} ({fav_prob:.1%})")
        print(f"  Spread: {game['home_team']} {pred['spread']:+.1f}")
    
    # Save
    df = pd.DataFrame(predictions)
    output_path = f"data/processed/week{week}_predictions_{season}.csv"
    df.to_csv(output_path, index=False)
    
    print("\n" + "="*70)
    print(f"Saved: {output_path}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Predict NFL games for a week')
    parser.add_argument('--week', type=int, required=True, help='Week number')
    parser.add_argument('--season', type=int, default=2025, help='Season')
    parser.add_argument(
        '--config',
        default='experiments/configs/elo_baseline.yaml',
        help='Config file to use'
    )
    
    args = parser.parse_args()
    predict_week(args.config, args.season, args.week)


if __name__ == "__main__":
    main()