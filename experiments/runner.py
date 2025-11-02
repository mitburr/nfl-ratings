"""Unified experiment runner with config-driven execution."""

import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List

from src.models.elo import EloPredictor
from src.models.vector_enhanced import VectorEnhancedPredictor
from src.evaluation.evaluator import ModelEvaluator
from src.utils.config import load_config, apply_overrides, validate_config
from src.utils.logging import setup_logging
from experiments.tracker import ExperimentTracker

logger = logging.getLogger(__name__)


def create_model(config: dict):
    """Create model instance from config.
    
    Args:
        config: Validated configuration dictionary
        
    Returns:
        Model instance
    """
    model_type = config['model_type']
    
    if model_type == 'elo':
        return EloPredictor(config['parameters'])
    elif model_type == 'vectors':
        return VectorEnhancedPredictor(config['parameters'])
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_experiment(
    config_path: str,
    seasons: List[int],
    overrides: List[str] = None,
    rolling: bool = True
):
    """Run single experiment.
    
    Args:
        config_path: Path to config YAML
        seasons: List of seasons to evaluate
        overrides: List of config overrides
        rolling: Use rolling evaluation
    """
    # Load and validate config
    config = load_config(config_path)
    
    if overrides:
        config = apply_overrides(config, overrides)
    
    config = validate_config(config)
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"logs/experiment_{config['name']}_{timestamp}.log"
    setup_logging('INFO', log_file=log_file)
    
    logger.info("="*70)
    logger.info(f"Starting experiment: {config['name']}")
    logger.info(f"Config: {config}")
    logger.info("="*70)
    
    # Create model
    model = create_model(config)
    logger.info(f"Created model: {model.__class__.__name__}")
    
    # Evaluate
    evaluator = ModelEvaluator()
    tracker = ExperimentTracker()
    
    all_results = []
    
    for season in seasons:
        logger.info(f"Evaluating season {season}")
        
        results = evaluator.evaluate_season(
            model,
            season,
            rolling=rolling,
            metrics=config.get('metrics', ['accuracy', 'log_loss'])
        )
        
        if results is not None:
            all_results.append(results)
            
            # Log results
            logger.info(f"Season {season} results:")
            for metric, value in results.items():
                if metric not in ['model', 'season', 'by_week', 'predictions', 'rolling']:
                    logger.info(f"  {metric}: {value:.4f}")
    
    # Track experiment
    if all_results:
        experiment_id = tracker.log_experiment(
            name=config['name'],
            config=config,
            results=all_results
        )
        
        logger.info("="*70)
        logger.info(f"Experiment complete: ID {experiment_id}")
        logger.info(f"Results saved to database")
        logger.info("="*70)
        
        return experiment_id
    else:
        logger.warning("No results to save")
        return None


def run_comparison(
    config_paths: List[str],
    seasons: List[int],
    rolling: bool = True
):
    """Compare multiple configurations.
    
    Args:
        config_paths: List of config file paths
        seasons: Seasons to evaluate
        rolling: Use rolling evaluation
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    setup_logging('INFO', log_file=f"logs/comparison_{timestamp}.log")
    
    logger.info("="*70)
    logger.info(f"Running comparison of {len(config_paths)} configs")
    logger.info("="*70)
    
    models = []
    for config_path in config_paths:
        config = validate_config(load_config(config_path))
        model = create_model(config)
        models.append(model)
        logger.info(f"Loaded: {model.name}")
    
    # Compare
    evaluator = ModelEvaluator()
    comparison_df = evaluator.compare_models(models, seasons, rolling=rolling)
    
    # Save results
    output_path = f"experiments/results/comparison_{timestamp}.csv"
    comparison_df.to_csv(output_path, index=False)
    
    logger.info("="*70)
    logger.info("Comparison Results:")
    logger.info("="*70)
    print(comparison_df.to_string(index=False))
    
    logger.info(f"\nSaved to: {output_path}")
    
    return comparison_df


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Run NFL prediction experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single experiment
  python -m experiments.runner --config configs/elo_baseline.yaml --seasons 2024

  # Run with overrides
  python -m experiments.runner --config configs/elo_baseline.yaml --seasons 2024 2025 \\
    --override k_factor=25 home_advantage=50

  # Compare multiple configs
  python -m experiments.runner --compare configs/elo*.yaml --seasons 2024
        """
    )
    
    parser.add_argument(
        '--config',
        help='Path to config file'
    )
    parser.add_argument(
        '--compare',
        nargs='+',
        help='Compare multiple configs'
    )
    parser.add_argument(
        '--seasons',
        type=int,
        nargs='+',
        default=[2024],
        help='Seasons to evaluate'
    )
    parser.add_argument(
        '--override',
        nargs='*',
        help='Config overrides (key=value)'
    )
    parser.add_argument(
        '--no-rolling',
        action='store_true',
        help='Disable rolling evaluation'
    )
    
    args = parser.parse_args()
    
    if args.compare:
        # Comparison mode
        run_comparison(args.compare, args.seasons, rolling=not args.no_rolling)
    elif args.config:
        # Single experiment
        run_experiment(
            args.config,
            args.seasons,
            overrides=args.override,
            rolling=not args.no_rolling
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()