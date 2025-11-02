"""Configuration utilities for experiment management."""

import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path) as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded config from {config_path}")
    return config


def apply_overrides(config: Dict, overrides: List[str]) -> Dict:
    """Apply command-line overrides to config.
    
    Supports dot notation for nested keys: 'elo.k_factor=25'
    
    Args:
        config: Base configuration dictionary
        overrides: List of 'key=value' strings
        
    Returns:
        Modified configuration dictionary
    """
    for override in overrides:
        if '=' not in override:
            logger.warning(f"Invalid override format: {override}")
            continue
        
        key, value = override.split('=', 1)
        keys = key.split('.')
        
        # Navigate to target location
        target = config
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        
        # Set value with type inference
        try:
            parsed_value = yaml.safe_load(value)
            target[keys[-1]] = parsed_value
            logger.debug(f"Override: {key} = {parsed_value}")
        except Exception as e:
            logger.error(f"Failed to parse override {override}: {e}")
    
    return config


def validate_config(config: Dict) -> Dict:
    """Validate configuration dictionary.
    
    Args:
        config: Configuration to validate
        
    Returns:
        Validated configuration
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Required fields
    required = ['name', 'model_type', 'parameters']
    for field in required:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")
    
    # Model type validation
    valid_types = ['elo', 'vectors', 'ensemble']
    if config['model_type'] not in valid_types:
        raise ValueError(f"Invalid model_type: {config['model_type']}. "
                        f"Must be one of {valid_types}")
    
    # Parameter validation
    params = config['parameters']
    
    if 'k_factor' in params:
        k = params['k_factor']
        if not isinstance(k, (int, float)) or k <= 0:
            raise ValueError(f"k_factor must be positive number, got {k}")
    
    if 'home_advantage' in params:
        ha = params['home_advantage']
        if not isinstance(ha, (int, float)):
            raise ValueError(f"home_advantage must be number, got {ha}")
    
    # Vector config validation
    if config['model_type'] == 'vectors':
        vector_config = params.get('vector_config', {})
        
        if 'boost' in vector_config:
            b = vector_config['boost']
            if not isinstance(b, (int, float)) or b < 0:
                raise ValueError(f"boost must be non-negative number, got {b}")
        
        if 'weight' in vector_config:
            w = vector_config['weight']
            if not isinstance(w, (int, float)) or w < 0 or w > 1:
                raise ValueError(f"weight must be in [0,1], got {w}")
    
    logger.info(f"Config validated: {config['name']}")
    return config


def save_config(config: Dict, output_path: str):
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save config
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Saved config to {output_path}")