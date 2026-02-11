"""Transformer-based architectures for Learning to Rank.

This module implements context-aware rankers using Self-Attention mechanisms
to model document interactions within a single query group.
"""

from __future__ import annotations

from typing import Any, Dict

import keras
import tensorflow as tf

__all__ = [
    "TransformerRanker",
]


@keras.utils.register_keras_serializable(package="LTR")
class TransformerRanker(keras.Model):
    """
    Implements a Context-Aware Ranking model using a Transformer Encoder architecture.
    
    Architecture Summary:
    ---------------------
    Unlike "Pointwise" models (like standard ListNet) which score each document independently, 
    this model processes the entire list of documents simultaneously. It uses Self-Attention 
    to capture "Cross-Document Interactions," allowing the model to adjust the score of a 
    document based on the other documents present in the same query list.
    
    1. Input Projection: Projects raw features (batch, list_size, num_features) to 'head_dim * num_heads'.
    2. Multi-Head Self-Attention: Mechanisms to compare every document against every other document.
       - Includes a mask to ignore padding tokens.
    3. Residual Connections & Layer Normalization: For training stability.
    4. Feed-Forward Network (FFN): A 2-layer MLP applied to each position.
    5. Scoring Head: Projects the contextualized vector down to a scalar relevance score.
    
    References:
    -----------
    1. Pobrotyn et al. (2020). "Context-Aware Learning to Rank with Self-Attention".
       (Proposed replacing dense layers with Self-Attention for ranking).
       
    2. Pasumarthi et al. (2019). "Permutation Equivariant Document Interaction Network for Neural Learning to Rank".
       (Formalized the mathematics of using Self-Attention for ranking and proved permutation equivariance).
       
    3. Pang et al. (2020). "SetRank: Learning a Permutation-Invariant Ranking Model for Information Retrieval".
       (SIGIR 2020 paper detailing the importance of self-attention for variable-sized list ranking).
    """

    def __init__(
        self,
        num_features: int,
        num_heads: int = 4,
        head_dim: int = 32,
        ff_dim: int = 128,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        """Initializes the TransformerRanker.

        Args:
            num_features: Total number of features per document.
            num_heads: Number of attention heads.
            head_dim: Dimensionality of each attention head.
            ff_dim: Dimensionality of the feed-forward layer.
            dropout_rate: Dropout probability for regularization.
            **kwargs: Standard Keras Model arguments.
        """
        super().__init__(**kwargs)
        self.num_features = num_features
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.input_norm = keras.layers.LayerNormalization(epsilon=1e-6)
        self.projection = keras.layers.Dense(num_heads * head_dim)
        
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=head_dim, 
            dropout=dropout_rate
        )
        self.att_norm = keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.ffn_dense1 = keras.layers.Dense(ff_dim, activation="relu")
        self.ffn_dense2 = keras.layers.Dense(num_heads * head_dim)
        self.ffn_norm = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = keras.layers.Dropout(dropout_rate)
        
        self.score_layer = keras.layers.Dense(1)

    def build(self, input_shape: tf.TensorShape):
        """Standard Keras build method."""
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        # input_shape is (batch_size, num_docs, num_features)
        # our output is (batch_size, num_docs)
        return (input_shape[0], input_shape[1])

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass for the Transformer ranker.

        Args:
            inputs: Input features of shape [batch, list_size, num_features].
            training: Boolean indicating if the model is in training mode.

        Returns:
            Computed scores of shape [batch, list_size].
        """
        # 1. Create Mask (assuming padding is 0.0)
        mask = tf.math.reduce_any(tf.math.not_equal(inputs, 0.0), axis=-1)
        attn_mask = mask[:, tf.newaxis, :]
        
        # 2. Pre-process
        x = self.input_norm(inputs)
        x = self.projection(x)
        
        # 3. Self-Attention
        att_out = self.attention(
            query=x, 
            value=x, 
            key=x, 
            attention_mask=attn_mask, 
            training=training
        )
        x = self.att_norm(x + att_out)
        
        # 4. Feed Forward
        ffn_out = self.ffn_dense1(x)
        ffn_out = self.dropout(ffn_out, training=training)
        ffn_out = self.ffn_dense2(ffn_out)
        x = self.ffn_norm(x + ffn_out)
        
        # 5. Score
        scores = self.score_layer(x)
        scores = tf.squeeze(scores, axis=-1)

        # 6. Apply Padding Mask to Scores
        return tf.where(mask, scores, tf.fill(tf.shape(scores), -1e10))

    def get_config(self) -> Dict[str, Any]:
        """Returns the config for serialization."""
        config = super().get_config()
        config.update({
            "num_features": self.num_features,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config