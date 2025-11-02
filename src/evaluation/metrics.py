"""Metrics for evaluating prediction models."""

import numpy as np
import pandas as pd
from typing import Callable, Dict


def accuracy(predictions: pd.DataFrame) -> float:
    """Calculate prediction accuracy."""
    return predictions['correct'].mean()


def log_loss(predictions: pd.DataFrame) -> float:
    """Calculate log loss (cross-entropy)."""
    return predictions['log_loss'].mean()


def brier_score(predictions: pd.DataFrame) -> float:
    """Calculate Brier score."""
    return np.mean(
        (predictions['predicted_prob'] - predictions['actual_result']) ** 2
    )


def calibration_error(predictions: pd.DataFrame, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (ECE)."""
    bins = np.linspace(0, 1, n_bins + 1)
    errors = []
    
    for i in range(n_bins):
        mask = (predictions['predicted_prob'] >= bins[i]) & \
               (predictions['predicted_prob'] < bins[i + 1])
        
        if mask.sum() > 0:
            avg_pred = predictions.loc[mask, 'predicted_prob'].mean()
            avg_actual = predictions.loc[mask, 'actual_result'].mean()
            bin_weight = mask.sum() / len(predictions)
            errors.append(abs(avg_pred - avg_actual) * bin_weight)
    
    return sum(errors)


# Registry of available metrics
METRICS: Dict[str, Callable] = {
    'accuracy': accuracy,
    'log_loss': log_loss,
    'brier_score': brier_score,
    'calibration_error': calibration_error
}


def compute_metrics(
    predictions: pd.DataFrame,
    metric_names: list = None
) -> Dict[str, float]:
    """Compute multiple metrics on predictions.
    
    Args:
        predictions: DataFrame with 'predicted_prob', 'actual_result', 'correct', 'log_loss'
        metric_names: List of metric names to compute (default: all)
        
    Returns:
        Dictionary mapping metric names to values
    """
    if metric_names is None:
        metric_names = list(METRICS.keys())
    
    results = {}
    for name in metric_names:
        if name in METRICS:
            results[name] = METRICS[name](predictions)
        else:
            raise ValueError(f"Unknown metric: {name}")
    
    return results