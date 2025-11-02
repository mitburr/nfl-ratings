"""Experiment tracking with SQLite."""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Track experiments in SQLite database."""
    
    def __init__(self, db_path: str = "experiments/results/experiments.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    config TEXT NOT NULL,
                    seasons TEXT NOT NULL,
                    avg_accuracy REAL,
                    avg_log_loss REAL,
                    avg_brier_score REAL,
                    total_games INTEGER
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS season_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER NOT NULL,
                    season INTEGER NOT NULL,
                    accuracy REAL,
                    log_loss REAL,
                    brier_score REAL,
                    total_games INTEGER,
                    predictions_path TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                )
            """)
    
    def log_experiment(
        self,
        name: str,
        config: Dict,
        results: List[Dict]
    ) -> int:
        """Log experiment results.
        
        Args:
            name: Experiment name
            config: Configuration dictionary
            results: List of result dicts (one per season)
            
        Returns:
            Experiment ID
        """
        with sqlite3.connect(self.db_path) as conn:
            # Calculate aggregates
            seasons = [r['season'] for r in results]
            avg_accuracy = sum(r.get('accuracy', 0) for r in results) / len(results)
            avg_log_loss = sum(r.get('log_loss', 0) for r in results) / len(results)
            avg_brier = sum(r.get('brier_score', 0) for r in results) / len(results)
            total_games = sum(r.get('total_games', 0) for r in results)
            
            # Insert experiment
            cursor = conn.execute("""
                INSERT INTO experiments 
                (name, timestamp, config, seasons, avg_accuracy, avg_log_loss, 
                 avg_brier_score, total_games)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                name,
                datetime.now().isoformat(),
                json.dumps(config),
                json.dumps(seasons),
                avg_accuracy,
                avg_log_loss,
                avg_brier,
                total_games
            ))
            
            experiment_id = cursor.lastrowid
            
            # Insert season results
            for result in results:
                # Save predictions if present
                predictions_path = None
                if 'predictions' in result:
                    predictions_path = f"experiments/results/{experiment_id}_{result['season']}.csv"
                    result['predictions'].to_csv(predictions_path, index=False)
                
                conn.execute("""
                    INSERT INTO season_results
                    (experiment_id, season, accuracy, log_loss, brier_score, 
                     total_games, predictions_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    experiment_id,
                    result['season'],
                    result.get('accuracy'),
                    result.get('log_loss'),
                    result.get('brier_score'),
                    result.get('total_games'),
                    predictions_path
                ))
            
            logger.info(f"Logged experiment {experiment_id}: {name}")
            return experiment_id
    
    def get_experiments(
        self,
        name_pattern: Optional[str] = None,
        limit: int = 50
    ) -> pd.DataFrame:
        """Query experiment history.
        
        Args:
            name_pattern: Filter by name (SQL LIKE pattern)
            limit: Maximum number of results
            
        Returns:
            DataFrame with experiment history
        """
        query = "SELECT * FROM experiments"
        params = []
        
        if name_pattern:
            query += " WHERE name LIKE ?"
            params.append(f"%{name_pattern}%")
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql(query, conn, params=params)
    
    def get_experiment_details(self, experiment_id: int) -> Dict:
        """Get detailed results for an experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Dictionary with experiment and season results
        """
        with sqlite3.connect(self.db_path) as conn:
            # Get experiment
            exp_df = pd.read_sql(
                "SELECT * FROM experiments WHERE id = ?",
                conn,
                params=(experiment_id,)
            )
            
            if len(exp_df) == 0:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            # Get season results
            season_df = pd.read_sql(
                "SELECT * FROM season_results WHERE experiment_id = ?",
                conn,
                params=(experiment_id,)
            )
            
            return {
                'experiment': exp_df.iloc[0].to_dict(),
                'season_results': season_df
            }
    
    def compare_experiments(self, experiment_ids: List[int]) -> pd.DataFrame:
        """Compare multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs
            
        Returns:
            Comparison DataFrame
        """
        placeholders = ','.join('?' * len(experiment_ids))
        query = f"""
            SELECT 
                e.id,
                e.name,
                e.timestamp,
                e.avg_accuracy,
                e.avg_log_loss,
                e.avg_brier_score,
                e.total_games
            FROM experiments e
            WHERE e.id IN ({placeholders})
            ORDER BY e.avg_accuracy DESC
        """
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql(query, conn, params=experiment_ids)