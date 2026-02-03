"""Loss functions for Learning to Rank.

This module implements listwise loss functions that optimize the permutation 
probability of document lists, specifically focusing on ListNet-style 
Top-1 probability cross-entropy.
"""

from __future__ import annotations

import keras
import tensorflow as tf

__all__ = ["ListwiseSoftmaxLoss"]


@keras.saving.register_keras_serializable(package="LTR")
class ListwiseSoftmaxLoss(keras.losses.Loss):
    """Implements the ListNet Listwise Softmax Loss.

    This loss treats the relevance labels of a query group as a probability 
    distribution and minimizes the Cross-Entropy between the true distribution 
    (from labels) and the predicted distribution (from model scores).

    Attributes:
        padding_value: The value in y_true used to identify padded documents.
        temperature: A scaling factor for the softmax distribution. Higher 
            values result in "softer" distributions.
    """

    def __init__(
        self, 
        padding_value: float = -1.0, 
        temperature: float = 1.0, 
        name: str = "listwise_softmax_loss",
        **kwargs
    ):
        """Initializes the loss instance.

        Args:
            padding_value: Float indicating padded elements in y_true.
            temperature: Float for scaling logits before softmax.
            name: Optional name for the loss instance.
            **kwargs: Standard Keras loss keyword arguments.
        """
        super().__init__(name=name, **kwargs)
        self.padding_value = padding_value
        self.temperature = temperature

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Computes the listwise cross-entropy loss.

        Args:
            y_true: Ground truth relevance labels of shape [batch, list_size].
            y_pred: Predicted scores of shape [batch, list_size].

        Returns:
            A scalar float32 Tensor representing the mean loss over the batch.
        """
        # 1. Create mask for padded values
        # Shape: [batch, list_size]
        mask = tf.not_equal(y_true, self.padding_value)
        
        # 2. Prepare predicted distribution (Logits)
        # We use a very small value (-1e10) for padding to ensure exp(x) -> 0
        y_pred_masked = tf.where(mask, y_pred / self.temperature, -1e10)
        
        # 3. Prepare true distribution
        # Softmax over labels converts relevance scores into a probability distribution.
        # Documents with higher relevance labels get a higher probability mass.
        y_true_masked = tf.where(mask, y_true / self.temperature, -1e10)
        p_true = tf.nn.softmax(y_true_masked, axis=-1)
        
        # 4. Numerically stable Cross Entropy
        # Using log_softmax is more stable than log(softmax(x))
        log_p_pred = tf.nn.log_softmax(y_pred_masked, axis=-1)
        
        # Cross-entropy formula: -sum(P_true * log(P_pred))
        # Masking is implicit here because p_true is ~0 for padded documents.
        loss = -tf.reduce_sum(p_true * log_p_pred, axis=-1)
        
        return tf.reduce_mean(loss)

    def get_config(self):
        """Returns the config dictionary for serialization."""
        config = super().get_config()
        config.update({
            "padding_value": self.padding_value,
            "temperature": self.temperature,
        })
        return config