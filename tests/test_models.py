import pytest
import tensorflow as tf
import numpy as np
from learning_to_rank.models.listnet import ListNet, ResListNet

@pytest.mark.parametrize("model_class", [ListNet, ResListNet])
def test_model_output_shape(model_class):
    """Verify that models accept 3D input and return 2D scores."""
    num_features = 10
    batch_size = 4
    num_docs = 5
    
    model = model_class(num_features=num_features, hidden_units=[16, 8])
    inputs = tf.random.uniform((batch_size, num_docs, num_features))
    
    outputs = model(inputs, training=False)
    
    assert outputs.shape == (batch_size, num_docs)

@pytest.mark.parametrize("model_class", [ListNet, ResListNet])
def test_model_serialization(model_class):
    """Ensure models can be reconstructed from their configuration."""
    num_features = 136
    hidden_units = [32, 16]
    
    model = model_class(num_features=num_features, hidden_units=hidden_units)
    config = model.get_config()
    
    new_model = model_class.from_config(config)
    
    assert new_model.num_features == num_features
    assert new_model.hidden_units == hidden_units
    assert len(new_model.dense_layers) == len(hidden_units)

def test_resnet_skip_connection():
    """Verify ResListNet can handle increasing/decreasing hidden dimensions."""
    # Test that it doesn't crash when projecting from 10 features to 32 units
    model = ResListNet(num_features=10, hidden_units=[32, 16])
    inputs = tf.random.uniform((1, 5, 10))
    # Should run without error due to projection layers
    outputs = model(inputs)
    assert outputs.shape == (1, 5)