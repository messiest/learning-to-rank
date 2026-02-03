"""Evaluation metrics for Learning to Rank.

This module implements ranking-specific metrics, primarily NDCG, using 
TensorFlow operations optimized for query-grouped batches.
"""

from __future__ import annotations

import keras
import tensorflow as tf

__all__ = ["NDCGMetric"]


@keras.saving.register_keras_serializable(package="LTR")
class NDCGMetric(keras.metrics.Metric):
    """Computes Normalized Discounted Cumulative Gain (NDCG) at K.

    NDCG measures the quality of a ranking by comparing the DCG of the predicted 
    order against the Ideal DCG (IDCG) obtained by sorting the ground-truth labels.

    Attributes:
        top_k: The rank position at which to evaluate (e.g., NDCG@5).
        padding_value: The value in y_true used to identify padded documents.
        total_ndcg: Accumulated sum of NDCG scores across all queries.
        count: Total number of queries processed.
    """

    def __init__(
        self, 
        top_k: int = 5, 
        padding_value: float = -1.0, 
        name: str = "ndcg", 
        **kwargs
    ):
        """Initializes the NDCG metric instance.

        Args:
            top_k: The cutoff for the ranking list.
            padding_value: Float indicating padded elements in y_true.
            name: String name of the metric instance.
            **kwargs: Standard Keras metric keyword arguments.
        """
        super().__init__(name=name, **kwargs)
        self.top_k = top_k
        self.padding_value = padding_value
        self.total_ndcg = self.add_weight(name="total_ndcg", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: Optional[tf.Tensor] = None):
        """Accumulates the NDCG scores for a batch of queries.

        Args:
            y_true: Ground truth relevance labels of shape [batch, list_size].
            y_pred: Predicted scores of shape [batch, list_size].
            sample_weight: Optional weighting for each query.
        """
        # 1. Masking: Ensure padding doesn't interfere with top_k sorting
        mask = tf.not_equal(y_true, self.padding_value)
        y_pred_masked = tf.where(mask, y_pred, -1e10)
        y_true_masked = tf.where(mask, y_true, 0.0) # Padding adds 0 gain
        
        batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
        list_size = tf.shape(y_true)[1]
        k = tf.minimum(self.top_k, list_size)

        def _apply_discount(ordered_labels: tf.Tensor) -> tf.Tensor:
            """Helper to apply the DCG formula: sum((2^rel - 1) / log2(rank + 1))."""
            # Gain: 2^relevance - 1
            gains = tf.math.pow(2.0, ordered_labels) - 1.0
            
            # Discount: 1 / log2(rank + 1)
            ranks = tf.range(1, k + 1, dtype=tf.float32)
            discounts = tf.math.log(ranks + 1.0) / tf.math.log(2.0)
            
            return tf.reduce_sum(gains / discounts, axis=-1)

        # 2. Compute IDCG: Perfect ranking based on true labels
        idcg_labels, _ = tf.math.top_k(y_true_masked, k=k)
        idcg = _apply_discount(idcg_labels)

        # 3. Compute DCG: True labels ordered by predicted scores
        _, top_indices = tf.math.top_k(y_pred_masked, k=k)
        
        # Batch indexing for gather_nd
        batch_ids = tf.range(tf.shape(y_true)[0])[:, tf.newaxis]
        batch_ids = tf.tile(batch_ids, [1, k])
        full_indices = tf.stack([batch_ids, top_indices], axis=-1)
        
        dcg_labels = tf.gather_nd(y_true_masked, full_indices)
        dcg = _apply_discount(dcg_labels)

        # 4. Result calculation
        batch_ndcg = tf.math.divide_no_nan(dcg, idcg)
        
        self.total_ndcg.assign_add(tf.reduce_sum(batch_ndcg))
        self.count.assign_add(batch_size)

    def result(self) -> tf.Tensor:
        """Returns the mean NDCG across all processed batches."""
        return tf.math.divide_no_nan(self.total_ndcg, self.count)

    def reset_state(self):
        """Resets the metric weights to zero."""
        self.total_ndcg.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        """Returns the config dictionary for serialization."""
        config = super().get_config()
        config.update({
            "top_k": self.top_k,
            "padding_value": self.padding_value,
        })
        return config