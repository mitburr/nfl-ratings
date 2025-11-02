"""Caching utilities for model states."""

import hashlib
import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class EloCache:
    """Cache Elo ratings to avoid redundant calculations."""
    
    def __init__(self):
        self.cache: Dict[str, Any] = {}
    
    def _hash_config(self, config: Dict) -> str:
        """Create hash of relevant config parameters.
        
        Only includes parameters that affect Elo calculation.
        """
        relevant = {
            'k_factor': config.get('k_factor'),
            'home_advantage': config.get('home_advantage'),
            'initial_rating': config.get('initial_rating'),
            'mov_multiplier': config.get('mov_multiplier')
        }
        
        config_str = json.dumps(relevant, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def _make_key(self, season: int, through_week: Optional[int], config: Dict) -> str:
        """Create cache key."""
        config_hash = self._hash_config(config)
        week_str = str(through_week) if through_week is not None else 'all'
        return f"{season}_{week_str}_{config_hash}"
    
    def get(
        self,
        season: int,
        through_week: Optional[int],
        config: Dict
    ) -> Optional[Dict]:
        """Retrieve cached state if available.
        
        Args:
            season: Season year
            through_week: Week number (None for full season)
            config: Model configuration
            
        Returns:
            Cached state dict or None if not cached
        """
        key = self._make_key(season, through_week, config)
        
        if key in self.cache:
            logger.debug(f"Cache hit: {key}")
            return self.cache[key]
        
        logger.debug(f"Cache miss: {key}")
        return None
    
    def save(
        self,
        season: int,
        through_week: Optional[int],
        config: Dict,
        state: Dict
    ):
        """Save state to cache.
        
        Args:
            season: Season year
            through_week: Week number (None for full season)
            config: Model configuration
            state: State dictionary to cache
        """
        key = self._make_key(season, through_week, config)
        self.cache[key] = state
        logger.debug(f"Cached: {key}")
    
    def clear(self):
        """Clear all cached data."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def size(self) -> int:
        """Get number of cached entries."""
        return len(self.cache)