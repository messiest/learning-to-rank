"""Ensemble architectures for Learning to Rank serving.

This module provides classes for combining multiple trained ranking models
into a single inference graph using various fusion techniques.
"""

from __future__ import annotations

import keras
import tensorflow as tf


__all__ = ["EnsembleModel", "RRFEnsembleModel"]

@keras.saving.register_keras_serializable(package="LTR")
class EnsembleModel(keras.Model):
    def __init__(self, model_paths: list[str], **kwargs):
        """
        A Keras Model that ensembles multiple SavedModels via averaging.
        
        Args:
            model_paths (list of str): Paths to the SavedModels on disk.
            **kwargs: Standard arguments for keras.Model (name, etc.)
        """
        super().__init__(**kwargs)
        self.model_paths = model_paths
        
        # We perform loading in __init__ so the models are ready immediately.
        # Note: We do not track these as Keras Layers because they are raw SavedModels.
        # If you need them to be trainable or strictly tracked, use tf.keras.layers.TFSMLayer (TF 2.13+).
        self.models = []
        self.output_keys = []

        print(f"--- Building Ensemble from {len(model_paths)} models ---")
        for i, path in enumerate(model_paths):
            print(f"Loading: {path}")
            loaded = tf.saved_model.load(path)
            self.models.append(loaded)
            
            # Dynamic Key Detection
            # We assume the default signature is what we want
            sig = loaded.signatures['serving_default']
            
            # Grab the first output key (e.g., 'dense', 'score', 'output_0')
            key = list(sig.structured_outputs.keys())[0]
            self.output_keys.append(key)
            print(f" -> Model {i+1} Output Key detected: '{key}'")

    def call(self, inputs):
        """
        Standard Keras forward pass.
        Replaces the old 'serve' method.
        """
        predictions_list = []

        for i, model in enumerate(self.models):
            # 1. Get the serving signature
            infer = model.signatures['serving_default']
            
            # 2. Run Inference
            # Note: SavedModel signatures expect distinct arguments or a dictionary.
            # If inputs is a Tensor, we pass it directly.
            result_dict = infer(inputs)
            
            # 3. Extract Score using the pre-detected key
            key = self.output_keys[i]
            score = result_dict[key]
            predictions_list.append(score)

        # 4. Average
        # Stack shape: [Num_Models, Batch_Size, 1]
        stacked_predictions = tf.stack(predictions_list)
        avg_score = tf.reduce_mean(stacked_predictions, axis=0)
        
        return avg_score

    def get_config(self):
        """
        Enables serialization. 
        Keras uses this to reconstruct the model logic (but not the sub-models' weights directly).
        When reloading, it will attempt to load paths from 'model_paths'.
        """
        config = super().get_config()
        config.update({
            "model_paths": self.model_paths
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Optional: Custom reconstruction logic if needed.
        Default implementation is usually sufficient, but explicit is safer.
        """
        return cls(**config)


@keras.saving.register_keras_serializable(package="LTRServing")
class RRFEnsembleModel(keras.Model):
    """Combines multiple ranking models using Reciprocal Rank Fusion (RRF).

    RRF combines the rankings from multiple models by summing the reciprocal
    of the document's rank in each model. It is robust to different output 
    scales and focuses on the consensus of document order.

    The score for a document d is calculated as:
        score(d) = sum_{m in models} (1 / (k + rank_m(d)))

    Attributes:
        model_paths: A list of paths to saved Keras models (.keras or .h5).
        k: A smoothing constant (standardly 60) that mitigates the impact 
            of high-ranking outliers.
    """

    def __init__(
        self, 
        model_paths: list[str], 
        k: int = 60, 
        **kwargs
    ) -> None:
        """Initializes the RRFEnsembleModel.

        Args:
            model_paths: List of file paths to the models to be ensembled.
            k: The RRF smoothing constant. Defaults to 60.
            **kwargs: Standard Keras model keyword arguments.
        """
        super().__init__(**kwargs)
        self.model_paths = model_paths
        self.k = k
        # Load models eagerly during initialization
        self.ranking_models = [keras.models.load_model(p) for p in model_paths]

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Performs forward pass to compute RRF scores.

        Args:
            inputs: A 3D tensor of shape [batch_size, num_docs, num_features].
            training: Boolean indicating if the model is in training mode.
                Always forced to False for sub-models during inference.

        Returns:
            A 2D tensor of shape [batch_size, num_docs] containing the 
            aggregated RRF scores.
        """
        # List to accumulate (1 / (k + rank)) for each model
        all_rrf_contributions = []

        for model in self.ranking_models:
            # 1. Generate raw scores from the sub-model
            # Shape: [batch_size, num_docs]
            raw_scores = model(inputs, training=False)

            # 2. Determine ranks using double argsort
            # Sorting descending (highest score = rank 1)
            # First argsort: Indices that would sort the scores
            # Second argsort: The rank of each element in the original array
            indices = tf.argsort(raw_scores, axis=-1, direction='DESCENDING')
            ranks = tf.argsort(indices, axis=-1)

            # 3. Apply the RRF formula
            # We add 1 to ranks because tf.argsort is 0-indexed (0 to num_docs-1)
            # RRF Score = 1.0 / (k + rank + 1)
            rank_contribution = 1.0 / (
                tf.cast(self.k, tf.float32) + tf.cast(ranks + 1, tf.float32)
            )
            all_rrf_contributions.append(rank_contribution)

        # 4. Aggregation: Sum contributions across all models
        # Shape: [batch_size, num_docs]
        ensemble_scores = tf.add_n(all_rrf_contributions)

        return ensemble_scores

    def get_config(self) -> dict:
        """Returns the configuration of the model for serialization.

        Returns:
            A dictionary containing the model's parameters.
        """
        config = super().get_config()
        config.update({
            "model_paths": self.model_paths,
            "k": self.k,
        })
        return config

    @classmethod
    def from_config(cls, config: dict) -> RRFEnsembleModel:
        """Creates an instance of the model from its config.

        Args:
            config: Configuration dictionary.

        Returns:
            A new instance of RRFEnsembleModel.
        """
        return cls(**config)