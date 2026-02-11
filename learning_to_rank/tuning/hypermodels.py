"""HyperModel definitions for Keras Tuner.

This module provides HyperModel classes that define the search space and 
compilation logic for ListNet, ResListNet, and TransformerRanker.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import keras
import keras_tuner as kt
import tensorflow as tf

from learning_to_rank.config import NUM_FEATURES
from learning_to_rank.losses import ListwiseSoftmaxLoss
from learning_to_rank.metrics import NDCGMetric
from learning_to_rank.models.listnet import ListNet, ResListNet
from learning_to_rank.models.transformer import TransformerRanker

__all__ = [
    "ListNetHyperModel",
    "ResListNetHyperModel",
    "TransformerRankerHyperModel",
]


def _build_functional_wrapper(
    model: keras.Model, num_features: int, name: str
) -> keras.Model:
    """Wraps a subclassed model in a Functional API container.
    
    This ensures Keras Tuner can correctly inspect the input/output shapes 
    and perform serialization without 'lazy' build issues.
    """
    inputs = keras.Input(shape=(None, num_features), name="input_features")
    outputs = model(inputs)
    return keras.Model(inputs=inputs, outputs=outputs, name=name)


class ListNetHyperModel(kt.HyperModel):
    """Search space for the standard ListNet MLP."""

    def __init__(self, num_features: int = NUM_FEATURES, name: Optional[str] = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_features = num_features

    def build(self, hp: kt.HyperParameters) -> keras.Model:
        """Builds the ListNet model with tuned hyperparameters."""
        num_layers = hp.Int('num_layers', min_value=1, max_value=4)
        base_units = hp.Int('base_units', min_value=32, max_value=256, step=32)
        dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
        
        hidden_units = [
            max(16, base_units // (2**i)) for i in range(num_layers)
        ]

        internal_model = ListNet(
            hidden_units=hidden_units, 
            dropout_rate=dropout_rate
        )
        
        model = _build_functional_wrapper(
            internal_model, self.num_features, "ListNet_Tuner"
        )
        
        lr = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
            loss=ListwiseSoftmaxLoss(padding_value=-1.0),
            metrics=[NDCGMetric(top_k=5, padding_value=-1.0)]
        )
        return model


class ResListNetHyperModel(kt.HyperModel):
    """Search space for the Deep Residual Ranker."""

    def __init__(self, num_features: int = NUM_FEATURES, name: Optional[str] = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_features = num_features

    def build(self, hp: kt.HyperParameters) -> keras.Model:
        """Builds the ResListNet model with tuned hyperparameters."""
        num_layers = hp.Int('num_layers', min_value=2, max_value=5)
        base_units = hp.Int('base_units', min_value=64, max_value=512, step=64)
        dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
        
        hidden_units = [
            max(16, base_units // (2**i)) for i in range(num_layers)
        ]

        internal_model = ResListNet(
            hidden_units=hidden_units,
            dropout_rate=dropout_rate
        )
        
        model = _build_functional_wrapper(
            internal_model, self.num_features, "ResListNet_Tuner"
        )
        
        lr = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
            loss=ListwiseSoftmaxLoss(padding_value=-1.0),
            metrics=[NDCGMetric(top_k=5, padding_value=-1.0)]
        )
        return model


class TransformerRankerHyperModel(kt.HyperModel):
    """Search space for the Context-Aware Transformer Ranker."""

    def __init__(self, num_features: int = NUM_FEATURES, name: Optional[str] = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_features = num_features

    def build(self, hp: kt.HyperParameters) -> keras.Model:
        """Builds the TransformerRanker model with tuned hyperparameters."""
        num_heads = hp.Choice('num_heads', values=[2, 4, 8])
        head_dim = hp.Int('head_dim', min_value=16, max_value=64, step=16)
        ff_dim = hp.Int('ff_dim', min_value=64, max_value=256, step=64)
        dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)

        internal_model = TransformerRanker(
            num_features=self.num_features,
            num_heads=num_heads,
            head_dim=head_dim,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate
        )
        
        model = _build_functional_wrapper(
            internal_model, self.num_features, "Transformer_Tuner"
        )
        
        lr = hp.Float('learning_rate', min_value=5e-5, max_value=5e-3, sampling='log')
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
            loss=ListwiseSoftmaxLoss(padding_value=-1.0),
            metrics=[NDCGMetric(top_k=5, padding_value=-1.0)]
        )
        return model