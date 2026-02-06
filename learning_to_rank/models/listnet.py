"""Model architectures for ListNet-based Learning to Rank.

This module provides standard Multi-Layer Perceptron (MLP) and Residual 
Network (ResNet) implementations optimized for listwise ranking tasks 
using TensorFlow/Keras.
"""

from __future__ import annotations

from typing import Any, Dict, List

import keras
import tensorflow as tf

from learning_to_rank.config import NUM_FEATURES
from learning_to_rank.layers import ResidualBlock

__all__ = [
    "ListNet",
    "ResListNet",
]


@keras.utils.register_keras_serializable(package="LTR")
class ListNet(keras.Model):
    """Standard Multi-Layer Perceptron (MLP) ranker.
    
    This model projects document features to a scalar score using a stack of 
    Dense layers. It is designed to process 3D inputs of shape 
    [batch_size, num_docs, num_features].

    Attributes:
        num_features: Number of input features per document.
        hidden_units: List of integers representing neurons in hidden layers.
        dropout_rate: Float representing the dropout probability.
    """

    def __init__(
        self, 
        hidden_units: List[int] = [128, 64, 32], 
        dropout_rate: float = 0.1, 
        **kwargs
    ):
        """Initializes the ListNet model.

        Args:
            num_features: Total number of features per document.
            hidden_units: Architecture of the hidden dense layers.
            dropout_rate: Fraction of units to drop during training.
            **kwargs: Standard Keras Model arguments.
        """
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        
        self.input_bn = keras.layers.BatchNormalization()
        
        self.dense_layers = [
            keras.layers.Dense(units, activation='relu') 
            for units in hidden_units
        ]
        self.dropout_layers = [
            keras.layers.Dropout(dropout_rate) 
            for _ in hidden_units
        ]
            
        self.score_layer = keras.layers.Dense(1, activation=None)

    def compute_output_shape(self, input_shape):
        """
        Input: (batch_size, num_docs, num_features)
        Output: (batch_size, num_docs)
        """
        # We return the batch size and document count, dropping the feature dim
        return (input_shape[0], input_shape[1])
    
    def build(self, input_shape):
        """Creates the weights of the model."""
        # input_shape is (batch_size, num_docs, num_features)
        # We grab the feature dimension: input_shape[-1]
        for units in self.hidden_units:
            self.dense_layers.append(keras.layers.Dense(units, activation="relu"))
        
        self.output_layer = keras.layers.Dense(1) # Linear output for ranking scores
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass of the model.

        Args:
            inputs: Tensor of shape [batch_size, num_docs, num_features].
            training: Boolean indicating whether the model is in training mode.

        Returns:
            Scores tensor of shape [batch_size, num_docs].
        """
        x = self.input_bn(inputs, training=training)
        
        for dense, dropout in zip(self.dense_layers, self.dropout_layers):
            x = dense(x)
            x = dropout(x, training=training)
                
        scores = self.score_layer(x)
        return tf.squeeze(scores, axis=-1)
    
    def compute_output_shape(self, input_shape):
        """
        Input: (batch_size, num_docs, num_features)
        Output: (batch_size, num_docs)
        """
        return (input_shape[0], input_shape[1])

    def get_config(self) -> Dict[str, Any]:
        """Returns the config for serialization."""
        config = super().get_config()
        config.update({
            "hidden_units": self.hidden_units,
            "dropout_rate": self.dropout_rate,
        })
        return config


@keras.utils.register_keras_serializable(package="LTR")
class ResListNet(keras.Model):
    """Deep Residual Ranker for Learning to Rank.
    
    Uses skip connections to enable deeper architectures without vanishing 
    gradients. Each layer projects the input to the current hidden dimension 
    to ensure the residual addition is mathematically valid.

    Structure: Input -> BN -> [Dense -> BN -> Dropout -> Skip] x N -> Score
    """

    def __init__(
        self, 
        hidden_units: List[int] = [256, 128, 64], 
        dropout_rate: float = 0.2, 
        **kwargs
    ):
        """Initializes the ResListNet model.

        Args:
            hidden_units: Architecture of the residual blocks.
            dropout_rate: Fraction of units to drop during training.
            **kwargs: Standard Keras Model arguments.
        """
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        
        self.input_bn = keras.layers.BatchNormalization()
        self.res_blocks = [
            ResidualBlock(units, dropout_rate) for units in hidden_units
        ]
        self.score_layer = keras.layers.Dense(1)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass with residual connections.

        Args:
            inputs: Tensor of shape [batch_size, num_docs, num_features].
            training: Boolean indicating whether the model is in training mode.

        Returns:
            Scores tensor of shape [batch_size, num_docs].
        """
        x = self.input_bn(inputs, training=training)
        
        for block in self.res_blocks:
            x = block(x, training=training)
                
        scores = self.score_layer(x)
        return tf.squeeze(scores, axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "hidden_units": self.hidden_units,
            "dropout_rate": self.dropout_rate,
        })
        return config
