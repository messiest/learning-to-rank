"""Model architectures for ListNet-based Learning to Rank.

This module provides standard Multi-Layer Perceptron (MLP) and Residual 
Network (ResNet) implementations optimized for listwise ranking tasks 
using TensorFlow/Keras.
"""

from __future__ import annotations

from typing import Any, Dict, List

import keras
import tensorflow as tf

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
        num_features: int, 
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
        self.num_features = num_features
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

    def get_config(self) -> Dict[str, Any]:
        """Returns the config for serialization."""
        config = super().get_config()
        config.update({
            "num_features": self.num_features,
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
        num_features: int, 
        hidden_units: List[int] = [256, 128, 64], 
        dropout_rate: float = 0.2, 
        **kwargs
    ):
        """Initializes the ResListNet model.

        Args:
            num_features: Total number of features per document.
            hidden_units: Architecture of the residual blocks.
            dropout_rate: Fraction of units to drop during training.
            **kwargs: Standard Keras Model arguments.
        """
        super().__init__(**kwargs)
        self.num_features = num_features
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        
        self.input_bn = keras.layers.BatchNormalization()
        
        self.dense_layers = []
        self.norm_layers = []
        self.dropout_layers = []
        self.projections = []
        
        for units in hidden_units:
            self.dense_layers.append(keras.layers.Dense(units, activation='relu'))
            self.norm_layers.append(keras.layers.BatchNormalization())
            self.dropout_layers.append(keras.layers.Dropout(dropout_rate))
            self.projections.append(keras.layers.Dense(units, use_bias=False))
            
        self.score_layer = keras.layers.Dense(1, activation=None)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass with residual connections.

        Args:
            inputs: Tensor of shape [batch_size, num_docs, num_features].
            training: Boolean indicating whether the model is in training mode.

        Returns:
            Scores tensor of shape [batch_size, num_docs].
        """
        x = self.input_bn(inputs, training=training)
        
        for i in range(len(self.dense_layers)):
            shortcut = self.projections[i](x)
            
            out = self.dense_layers[i](x)
            out = self.norm_layers[i](out, training=training)
            out = self.dropout_layers[i](out, training=training)
            
            x = out + shortcut
                
        scores = self.score_layer(x)
        return tf.squeeze(scores, axis=-1)

    def get_config(self) -> Dict[str, Any]:
        """Returns the config for serialization."""
        config = super().get_config()
        config.update({
            "num_features": self.num_features,
            "hidden_units": self.hidden_units,
            "dropout_rate": self.dropout_rate,
        })
        return config