"""Base predictor interface for all NFL prediction models."""

from abc import ABC, abstractmethod
from typing import Dict, Optional


class BasePredictor(ABC):
    """Abstract base class for prediction models.
    
    All prediction models must implement fit() and predict() methods.
    Serialization methods (get_state/load_state) are optional.
    """
    
    def __init__(self, config: Dict):
        """Initialize predictor with configuration.
        
        Args:
            config: Dictionary containing model parameters
        """
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, season: int, through_week: Optional[int] = None):
        """Train/update model through specified week.
        
        Args:
            season: Season year
            through_week: Last week to include (None = all weeks)
            
        Returns:
            self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, home_team: str, away_team: str) -> Dict:
        """Predict single game outcome.
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            
        Returns:
            Dictionary containing:
                - home_win_prob: float
                - away_win_prob: float
                - spread: float (positive = home favored)
                - confidence: float (0-1)
                - metadata: dict with model-specific info
        """
        pass
    
    def get_state(self) -> Dict:
        """Get model state for serialization (optional override)."""
        return {}
    
    def load_state(self, state: Dict):
        """Restore model from saved state (optional override)."""
        pass