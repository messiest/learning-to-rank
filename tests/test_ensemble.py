import pytest
import tensorflow as tf
import numpy as np
from unittest.mock import MagicMock, patch
from learning_to_rank.serving_model.ensemble import EnsembleModel

# Fixture to provide a mock signature
@pytest.fixture
def mock_signature_obj():
    sig = MagicMock()
    # Mocking the dictionary returned by a SavedModel signature
    sig.return_value = {"output_0": tf.constant([[[0.5], [0.5]]], dtype=tf.float32)}
    sig.structured_outputs = {"output_0": None}
    return sig

@patch("tensorflow.saved_model.load")
@patch("tensorflow.keras.models.load_model")
def test_ensemble_full_coverage(mock_keras_load, mock_tf_load, tmp_path, mock_signature_obj):
    """
    Surgically targets all branches in ensemble.py.
    Uses 'run_functions_eagerly' to ensure coverage sees inside tf.function.
    """
    # 1. Setup paths to trigger all branches in __init__
    keras_path = str(tmp_path / "model.keras")
    h5_path = str(tmp_path / "model.h5")
    sm_path = str(tmp_path / "saved_model_dir")
    
    # 2. Setup Mocks for __init__
    # Model 1: Generic SavedModel (has signatures)
    mock_sm = MagicMock(spec=tf.Module)
    mock_sm.signatures = {"serving_default": mock_signature_obj}
    
    # Model 2: Keras Model (Directly callable, no signatures, 2D output)
    mock_keras = MagicMock()
    del mock_keras.signatures
    mock_keras.return_value = tf.constant([[1.0, 1.0]], dtype=tf.float32)
    
    mock_tf_load.return_value = mock_sm
    mock_keras_load.side_effect = [mock_keras, mock_keras]

    # 3. Enable Eager Execution for the duration of this test
    tf.config.run_functions_eagerly(True)
    
    try:
        # This triggers the __init__ loop branches
        ensemble = EnsembleModel([keras_path, h5_path, sm_path])
        
        # 4. Trigger __call__ branches
        # This will now be visible to coverage because of run_functions_eagerly
        test_input = tf.random.uniform((1, 2, 136))
        results = ensemble(test_input)
        
        # Verify Averaging Logic
        # Model 1 gives 0.5 (squeezed from 3D), Model 2 gives 1.0. Avg = 0.75
        assert "scores" in results
        assert results["scores"].numpy() == pytest.approx(0.8333333)
        
    finally:
        # Always reset eager execution
        tf.config.run_functions_eagerly(False)

def test_ensemble_is_trackable():
    """Simple check for tf.Module property tracking."""
    ensemble = EnsembleModel([])
    assert isinstance(ensemble.sub_models, dict)