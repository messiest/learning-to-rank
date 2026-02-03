"""Utility functions for LTR interpretability and inference.

This module provides tools for computing feature importance, visualizing 
ranking results, and performing memory-efficient ensemble inference.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from learning_to_rank.losses import ListwiseSoftmaxLoss
from learning_to_rank.metrics import NDCGMetric
from learning_to_rank.models.listnet import ListNet, ResListNet
from learning_to_rank.models.transformer import TransformerRanker

__all__ = [
    "compute_permutation_importance",
    "plot_feature_importance",
    "predict_ensemble",
]

# ==============================================================================
# 1. INTERPRETABILITY
# ==============================================================================

def compute_permutation_importance(
    model: keras.Model,
    dataset: tf.data.Dataset,
    feature_names: Optional[List[str]] = None,
    batch_limit: int = 50
) -> Dict[int, float]:
    """Computes Permutation Feature Importance (PFI) using NDCG drop.

    Measures the importance of a feature by calculating the decrease in 
    NDCG when that feature's values are randomly shuffled across the 
    batch, breaking its relationship with the target.

    Args:
        model: Trained ranking model.
        dataset: tf.data.Dataset yielding (features, labels).
        feature_names: Optional list of strings for feature identification.
        batch_limit: Maximum number of batches to process for evaluation.

    Returns:
        A dictionary mapping feature indices to their calculated NDCG drop.

    Raises:
        ValueError: If the dataset is empty or batch_limit is zero.
    """
    # 1. Prepare Metric
    metric = NDCGMetric(top_k=5, padding_value=-1.0)
    
    # 2. Extract Data to Numpy
    features_list = []
    labels_list = []
    
    for x, y in dataset.take(batch_limit):
        features_list.append(x.numpy())
        labels_list.append(y.numpy())
        
    if not features_list:
        raise ValueError("Dataset is empty or batch_limit is 0.")

    x_val = np.concatenate(features_list, axis=0)
    y_val = np.concatenate(labels_list, axis=0)
    
    # 3. Baseline Performance
    baseline_preds = model.predict(x_val, batch_size=32, verbose=0)
    metric.update_state(y_val, baseline_preds)
    baseline_ndcg = metric.result().numpy()
    metric.reset_state()
    
    # 4. Iterate Features
    num_features = x_val.shape[-1]
    importances = {}
    
    for i in range(num_features):
        x_shuffled = x_val.copy()
        
        # Shuffle across both Batch and List dimensions to break interaction
        flat_col = x_shuffled[:, :, i].flatten()
        np.random.shuffle(flat_col)
        x_shuffled[:, :, i] = flat_col.reshape(x_shuffled.shape[0], x_shuffled.shape[1])
        
        shuffled_preds = model.predict(x_shuffled, batch_size=32, verbose=0)
        metric.update_state(y_val, shuffled_preds)
        shuffled_ndcg = metric.result().numpy()
        metric.reset_state()
        
        # Drop = Importance (Lower NDCG after shuffle = Higher Importance)
        importances[i] = float(baseline_ndcg - shuffled_ndcg)
            
    return importances


# ==============================================================================
# 2. VISUALIZATION
# ==============================================================================

def plot_feature_importance(
    importances: Dict[int, float],
    feature_names: Optional[List[str]] = None,
    top_n: int = 20,
    save_path: Optional[str] = None
) -> None:
    """Plots a horizontal bar chart of the top N most important features.

    Args:
        importances: Dictionary of {index: importance_score}.
        feature_names: List of strings corresponding to feature indices.
        top_n: Number of top features to visualize.
        save_path: File path to save the plot. If None, shows the plot.
    """
    sorted_idx = sorted(importances, key=importances.get, reverse=True)
    top_indices = sorted_idx[:top_n]
    top_scores = [importances[i] for i in top_indices]
    
    labels = (
        [feature_names[i] for i in top_indices] 
        if feature_names else [f"Feature {i}" for i in top_indices]
    )
        
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), top_scores, align='center', color='#3498db')
    plt.yticks(range(top_n), labels)
    plt.xlabel('NDCG Drop (Importance)')
    plt.title(f'Top {top_n} Features (Permutation Importance)')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()


# ==============================================================================
# 3. INFERENCE
# ==============================================================================

def predict_ensemble(
    model_paths: List[str], 
    dataset: tf.data.Dataset
) -> Optional[np.ndarray]:
    """Sequentially loads models and averages their predictions.

    Optimized for memory by loading and deleting one model at a time 
    from the Keras session.

    Args:
        model_paths: List of paths to .keras model files.
        dataset: tf.data.Dataset to perform inference on.

    Returns:
        Numpy array of averaged scores or None if loading fails.
    """
    custom_objs = {
        "ResListNet": ResListNet,
        "ListNet": ListNet,
        "TransformerRanker": TransformerRanker,
        "ListwiseSoftmaxLoss": ListwiseSoftmaxLoss,
        "NDCGMetric": NDCGMetric
    }
    
    aggregated_scores = None
    count = 0
    
    for path in model_paths:
        try:
            model = keras.models.load_model(path, custom_objects=custom_objs)
            scores = model.predict(dataset, verbose=1)
            
            if aggregated_scores is None:
                aggregated_scores = scores
            else:
                aggregated_scores += scores
            
            count += 1
            del model
            tf.keras.backend.clear_session()
            
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue
            
    return aggregated_scores / count if count > 0 else None