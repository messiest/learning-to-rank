"""Model layers for ListNet-based Learning to Rank."""

from __future__ import annotations

import keras


__all__ = ["ResidualBlock"]

@keras.utils.register_keras_serializable(package="LTR")
class ResidualBlock(keras.layers.Layer):
    """A single Residual Block with optional projection.
    
    Structure: Input -> [Dense -> BN -> Dropout] + Shortcut -> ReLU
    """
    def __init__(self, units: int, dropout_rate: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        
        self.dense = keras.layers.Dense(units)
        self.bn = keras.layers.BatchNormalization()
        self.dropout = keras.layers.Dropout(dropout_rate)
        self.activation = keras.layers.Activation("relu")
        self.projection = None # Initialized in build()

    def build(self, input_shape):
        """Creates a projection layer if the input dimension doesn't match units."""
        input_dim = input_shape[-1]
        if input_dim != self.units:
            self.projection = keras.layers.Dense(self.units, use_bias=False)
        super().build(input_shape)

    def call(self, inputs, training=False):
        # 1. Main Path
        x = self.dense(inputs)
        x = self.bn(x, training=training)
        x = self.dropout(x, training=training)
        
        # 2. Shortcut Path
        shortcut = inputs
        if self.projection is not None:
            shortcut = self.projection(inputs)
            
        # 3. Combine and Activate
        x = keras.layers.Add()([x, shortcut])
        return self.activation(x)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "dropout_rate": self.dropout_rate})
        return config