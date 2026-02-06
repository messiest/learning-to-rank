"""Unit tests for serving ensemble models."""

import os
import pytest
import numpy as np
import tensorflow as tf
import keras

from learning_to_rank.serving_model.ensemble import EnsembleModel, RRFEnsembleModel

# --- Fixtures for creating temporary models ---

@pytest.fixture
def mock_saved_model_path(tmp_path):
    """
    Creates a simple TF SavedModel that returns (input * multiplier).
    Used to test EnsembleModel (which loads raw SavedModels).
    """
    def create_model(multiplier, name):
        # Define a simple module
        class SimpleModule(tf.Module):
            def __init__(self, mult):
                self.mult = tf.constant(mult, dtype=tf.float32)

            @tf.function(input_signature=[tf.TensorSpec(shape=[None, 10], dtype=tf.float32)])
            def __call__(self, x):
                return {"output_0": x * self.mult}

        model_dir = tmp_path / name
        tf.saved_model.save(SimpleModule(multiplier), str(model_dir))
        return str(model_dir)

    # Return a factory function so tests can create multiple distinct models
    return create_model

@pytest.fixture
def mock_keras_model_path(tmp_path):
    """
    Creates a simple Keras model saved as .keras.
    Used to test RRFEnsembleModel (which loads Keras models).
    """
    def create_model(weights_val, name):
        # A model that outputs constant scores based on weights
        # We use a Dense layer with fixed weights to simulate ranking scores
        inputs = keras.Input(shape=(10,))
        # Initialize kernel to weights_val (broadcasted)
        outputs = keras.layers.Dense(
            5, # Output 5 docs/scores
            kernel_initializer=tf.keras.initializers.Constant(weights_val),
            bias_initializer='zeros'
        )(inputs)
        model = keras.Model(inputs, outputs)
        
        save_path = tmp_path / f"{name}.keras"
        model.save(save_path)
        return str(save_path)

    return create_model

@pytest.fixture
def input_data():
    """Standard input tensor (Batch Size=2, Features=10)."""
    return tf.constant(np.ones((2, 10)), dtype=tf.float32)

# ==============================================================================
# Tests for EnsembleModel (Averaging)
# ==============================================================================

def test_ensemble_initialization(mock_saved_model_path):
    """Test that EnsembleModel loads paths and detects keys correctly."""
    path1 = mock_saved_model_path(1.0, "m1")
    path2 = mock_saved_model_path(2.0, "m2")
    
    ensemble = EnsembleModel(model_paths=[path1, path2])
    
    assert len(ensemble.models) == 2
    assert len(ensemble.output_keys) == 2
    assert ensemble.output_keys[0] == "output_0"

def test_ensemble_averaging_logic(mock_saved_model_path, input_data):
    """
    Test that the ensemble correctly averages the outputs.
    Model 1 returns x * 1.0
    Model 2 returns x * 3.0
    Average should be x * 2.0
    """
    path1 = mock_saved_model_path(1.0, "m1")
    path2 = mock_saved_model_path(3.0, "m2")
    
    ensemble = EnsembleModel(model_paths=[path1, path2])
    
    # Run inference
    # input is all ones. 
    # m1 -> 1.0, m2 -> 3.0. Avg -> 2.0
    output = ensemble(input_data)
    
    expected = input_data * 2.0
    np.testing.assert_allclose(output.numpy(), expected.numpy(), atol=1e-5)

def test_ensemble_serialization(mock_saved_model_path):
    """Test get_config and from_config for EnsembleModel."""
    path = mock_saved_model_path(1.0, "m1")
    ensemble = EnsembleModel(model_paths=[path])
    
    config = ensemble.get_config()
    
    assert "model_paths" in config
    assert config["model_paths"] == [path]
    
    # Recreate
    new_ensemble = EnsembleModel.from_config(config)
    assert len(new_ensemble.models) == 1
    assert new_ensemble.model_paths == [path]

# ==============================================================================
# Tests for RRFEnsembleModel (Rank Fusion)
# ==============================================================================

def test_rrf_logic_manual_calculation(tmp_path):
    """
    Verify RRF math logic strictly.
    We create two fake models that output fixed scores to force specific rankings.
    """
    # Create two dummy Keras models
    # We will override their call methods dynamically to avoid file IO overhead 
    # and strictly control output scores for this specific math test.
    
    # Scenario: Batch Size 1, 3 Documents. k=1.
    # Model A Scores: [0.9, 0.8, 0.7] -> Ranks: [0, 1, 2] (Doc 0 is first)
    # Model B Scores: [0.1, 0.9, 0.5] -> Ranks: [2, 0, 1] (Doc 1 is first)
    
    class FakeModel(keras.Model):
        def __init__(self, scores):
            super().__init__()
            self.scores = tf.constant([scores], dtype=tf.float32) # [1, 3]
        def call(self, inputs, training=False):
            return self.scores

    # Save dummy files just so __init__ doesn't crash loading them
    dummy_path = tmp_path / "dummy.keras"
    keras.Model().save(dummy_path)
    
    # Instantiate RRF
    rrf = RRFEnsembleModel(model_paths=[str(dummy_path), str(dummy_path)], k=1)
    
    # Inject our fake models
    rrf.ranking_models = [
        FakeModel([0.9, 0.8, 0.7]), # Ranks: 0, 1, 2
        FakeModel([0.1, 0.9, 0.5])  # Ranks: 2, 0, 1
    ]
    
    # Expected RRF Calculation:
    # Formula: sum(1 / (k + rank + 1))  <-- Note: code adds 1 to 0-indexed rank
    
    # Doc 0:
    #   Model A Rank 0 -> 1/(1+0+1) = 0.5
    #   Model B Rank 2 -> 1/(1+2+1) = 0.25
    #   Sum = 0.75
    
    # Doc 1:
    #   Model A Rank 1 -> 1/(1+1+1) = 0.333...
    #   Model B Rank 0 -> 1/(1+0+1) = 0.5
    #   Sum = 0.833...
    
    # Doc 2:
    #   Model A Rank 2 -> 1/(1+2+1) = 0.25
    #   Model B Rank 1 -> 1/(1+1+1) = 0.333...
    #   Sum = 0.583...

    output = rrf(tf.zeros((1, 3, 5))) # Input shape doesn't matter for FakeModel
    output = output.numpy()[0]
    
    np.testing.assert_allclose(output[0], 0.75, atol=1e-4)
    np.testing.assert_allclose(output[1], 0.8333, atol=1e-4)
    np.testing.assert_allclose(output[2], 0.5833, atol=1e-4)

def test_rrf_initialization(mock_keras_model_path):
    """Test that RRFEnsembleModel loads Keras models correctly."""
    path1 = mock_keras_model_path(0.5, "k1")
    path2 = mock_keras_model_path(0.8, "k2")
    
    rrf = RRFEnsembleModel(model_paths=[path1, path2], k=60)
    
    assert len(rrf.ranking_models) == 2
    assert isinstance(rrf.ranking_models[0], keras.Model)
    assert rrf.k == 60

def test_rrf_serialization(mock_keras_model_path):
    """Test get_config and from_config for RRFEnsembleModel."""
    path = mock_keras_model_path(0.5, "k1")
    rrf = RRFEnsembleModel(model_paths=[path], k=100)
    
    config = rrf.get_config()
    
    assert config["k"] == 100
    assert config["model_paths"] == [path]
    
    # Recreate
    new_rrf = RRFEnsembleModel.from_config(config)
    assert new_rrf.k == 100
    assert len(new_rrf.ranking_models) == 1

def test_rrf_robustness_to_scale(tmp_path):
    """
    RRF should ignore the raw scale of scores.
    If Model A outputs [100, 0] and Model B outputs [0.001, 0.0001],
    and both imply Rank 0 > Rank 1, RRF should treat them identically.
    """
    class HighScaleModel(keras.Model):
        def call(self, x, training=False):
            return tf.constant([[1000.0, 0.0]]) # Doc 0 > Doc 1

    class LowScaleModel(keras.Model):
        def call(self, x, training=False):
            return tf.constant([[0.001, 0.0001]]) # Doc 0 > Doc 1

    # Initialize dummy file (just to satisfy __init__ loader)
    dummy_path = tmp_path / "dummy.keras"
    keras.Model().save(dummy_path)
    
    # --- Test 1: High Scale ---
    rrf_high = RRFEnsembleModel(model_paths=[str(dummy_path)], k=60)
    # Manually inject the high scale model
    rrf_high.ranking_models = [HighScaleModel()] 
    score_high = rrf_high(tf.zeros((1, 2, 1))).numpy()
    
    # --- Test 2: Low Scale ---
    # FIX: Instantiate a NEW model instead of modifying the built one
    rrf_low = RRFEnsembleModel(model_paths=[str(dummy_path)], k=60)
    # Manually inject the low scale model
    rrf_low.ranking_models = [LowScaleModel()]
    score_low = rrf_low(tf.zeros((1, 2, 1))).numpy()
    
    # Scores should be identical because ranks are identical
    np.testing.assert_allclose(score_high, score_low, atol=1e-6)