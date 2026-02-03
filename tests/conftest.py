"""Pytest fixtures for the Learning to Rank library."""

import pytest
import tensorflow as tf
import numpy as np

@pytest.fixture
def mock_ltr_data():
    """Returns a mock batch of ranking data [batch_size, list_size, num_features]."""
    batch_size = 2
    list_size = 3
    num_features = 4
    
    features = tf.random.uniform((batch_size, list_size, num_features))
    # Mix in some padding (all zeros) for the last doc of the first query
    features = tf.concat([
        features[:1, :2, :], 
        tf.zeros((1, 1, num_features)),
        features[1:, :, :]
    ], axis=1)
    
    labels = tf.constant([
        [2.0, 1.0, -1.0], # Query 1 (with padding)
        [0.0, 3.0, 1.0]   # Query 2 (full)
    ], dtype=tf.float32)
    
    return features, labels

@pytest.fixture
def dummy_model_path(tmp_path):
    """Creates a temporary directory for model saving tests."""
    d = tmp_path / "models"
    d.mkdir()
    return str(d)