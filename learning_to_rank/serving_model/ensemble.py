"""Module for ensembling ranking models for inference.

This module provides a container for multiple ranking models, allowing 
them to be executed as a single unit to produce averaged scores.
"""

from __future__ import annotations
from typing import List, Dict, Any

import tensorflow as tf


__all__ = ["EnsembleModel"]


class EnsembleModel(tf.Module):
    """A TensorFlow Module to ensemble multiple SavedModels or Keras models.
    
    This class loads sub-models into a trackable dictionary and uses a 
    tf.function to execute them in parallel (when possible) or sequence, 
    returning the mean of their predicted scores.
    
    Attributes:
        sub_models (Dict[str, tf.Module]): A trackable dictionary containing 
            the loaded ranking models.
    """

    def __init__(self, model_paths: List[str]):
        """Initializes the ensemble by loading models from provided paths.

        Args:
            model_paths: A list of file paths to .keras files or 
                SavedModel directories.
        """
        super(EnsembleModel, self).__init__()
        self.sub_models = {}
        
        for i, path in enumerate(model_paths):
            # Logic to handle .keras vs SavedModel directory
            if path.endswith('.keras') or path.endswith('.h5'):
                # Load the Keras model (requires Keras environment)
                model = tf.keras.models.load_model(path)
            else:
                # Load a standard SavedModel (pure TensorFlow)
                model = tf.saved_model.load(path)
                
            self.sub_models[f'model_{i}'] = model

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, 136], dtype=tf.float32)
    ])
    def __call__(self, x: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Performs ensemble inference on a batch of queries.

        Args:
            x: Input feature tensor of shape [batch_size, list_size, 136].

        Returns:
            A dictionary containing the "scores" tensor of shape 
            [batch_size, list_size].
        """
        predictions_list = []

        # We sort keys to ensure deterministic execution order
        for name in sorted(self.sub_models.keys()):
            model = self.sub_models[name]
            
            # Check if it's a generic SavedModel (has signatures) 
            # or a Keras object (callable)
            if hasattr(model, 'signatures'):
                infer = model.signatures['serving_default']
                # Dynamically find the output key to avoid 'output_0' vs 'dense' issues
                output_key = list(infer.structured_outputs.keys())[0]
                score = infer(x)[output_key]
            else:
                score = model(x)
            
            # Standardize shape to [Batch, List]
            if len(score.shape) == 3:
                score = tf.squeeze(score, axis=-1)
                
            predictions_list.append(score)

        # Average the predictions across all models
        stacked = tf.stack(predictions_list, axis=0) # [num_models, batch, list]
        avg_score = tf.reduce_mean(stacked, axis=0)
        
        return {"scores": avg_score}