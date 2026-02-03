"""Unit tests for the utils module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tensorflow as tf
from learning_to_rank.models.listnet import ListNet
from learning_to_rank.utils import (
    compute_permutation_importance,
    plot_feature_importance,
    predict_ensemble,
)


def test_permutation_importance_logic():
    """Verify that PFI identifies a highly predictive feature."""
    model = ListNet(num_features=2, hidden_units=[16, 8])
    
    # Create very distinct data: 
    # If feature 0 is high, label is high. If low, label is low.
    x = np.random.rand(200, 5, 2).astype(np.float32)
    # Target is ordinal: higher feature value = higher relevance
    y = np.digitize(x[:, :, 0], bins=[0.2, 0.4, 0.6, 0.8]).astype(np.float32)
    
    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(20)
    
    # Use a higher learning rate for the test to ensure it learns quickly
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss="mse")
    model.fit(ds, epochs=15, verbose=0)
    
    importances = compute_permutation_importance(model, ds, batch_limit=10)
    
    # Feature 0 drop should now be positive and greater than noise (Feature 1)
    assert importances[0] > importances[1]
    assert importances[0] > 0


def test_plot_feature_importance(tmp_path):
    """Test that the plotting function handles inputs and file saving correctly."""
    importances = {0: 0.5, 1: 0.2, 2: 0.1}
    feature_names = ["A", "B", "C"]
    save_file = tmp_path / "test_plot.png"

    # Use a non-interactive backend for matplotlib during tests
    import matplotlib
    matplotlib.use('Agg')
    
    # Test saving to disk
    plot_feature_importance(
        importances, 
        feature_names=feature_names, 
        top_n=2, 
        save_path=str(save_file)
    )
    
    assert save_file.exists()


@patch("keras.models.load_model")
def test_predict_ensemble(mock_load_model, mock_ltr_data):
    """Test the ensemble prediction utility with mocked model loading."""
    features, _ = mock_ltr_data
    # Convert mock features to a dataset
    ds = tf.data.Dataset.from_tensor_slices((features, features)).batch(1)
    
    # Create a mock model that returns specific scores
    mock_model = MagicMock()
    # model.predict returns a numpy array of shape (Batch, List)
    mock_model.predict.return_value = np.array([[0.8, 0.2, 0.5]])
    mock_load_model.return_value = mock_model
    
    model_paths = ["fake/path/1.keras", "fake/path/2.keras"]
    
    avg_scores = predict_ensemble(model_paths, ds)
    
    # Check that it attempted to load both models
    assert mock_load_model.call_count == 2
    # Check that scores were averaged correctly (mean of two identical mocks)
    np.testing.assert_allclose(avg_scores, np.array([[0.8, 0.2, 0.5]]))
    assert avg_scores.shape == (1, 3)


def test_permutation_importance_empty_dataset():
    """Verify ValueError is raised when an empty dataset is provided."""
    model = ListNet(num_features=2)
    empty_ds = tf.data.Dataset.from_tensor_slices(([], [])).batch(1)
    
    with pytest.raises(ValueError, match="Dataset is empty"):
        compute_permutation_importance(model, empty_ds)