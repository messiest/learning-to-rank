"""Pytest unit tests for model layers."""

import pytest
import numpy as np
import keras

from learning_to_rank.layers import ResidualBlock

# --- Fixtures ---
@pytest.fixture
def base_config():
    """Returns standard configuration for the tests."""
    return {
        "input_dim": 32,
        "output_dim": 64,
        "batch_size": 10,
        "dropout_rate": 0.3
    }

@pytest.fixture
def random_input(base_config):
    """Generates a random input batch."""
    return np.random.random((
        base_config["batch_size"], 
        base_config["input_dim"]
    )).astype("float32")

# --- Tests ---

def test_layer_config(base_config):
    """Test that the layer configuration is stored and retrieved correctly."""
    layer = ResidualBlock(
        units=base_config["output_dim"], 
        dropout_rate=base_config["dropout_rate"]
    )
    config = layer.get_config()
    
    assert config["units"] == base_config["output_dim"]
    assert config["dropout_rate"] == base_config["dropout_rate"]
    
    # Check inheritance from base Layer config
    assert "name" in config
    assert "trainable" in config

def test_forward_pass_no_projection(base_config):
    """
    Test forward pass when input_dim == units.
    The projection layer should remain None.
    """
    # Match input_dim (32) to units (32)
    units = base_config["input_dim"]
    layer = ResidualBlock(units=units)
    
    x = np.random.random((base_config["batch_size"], base_config["input_dim"])).astype("float32")
    
    # Run once to build
    y = layer(x)
    
    # Check output shape
    assert y.shape == (base_config["batch_size"], units)
    
    # Verify projection was NOT created
    assert layer.projection is None, "Projection should be None when input dim matches units."

def test_forward_pass_with_projection(base_config, random_input):
    """
    Test forward pass when input_dim != units.
    A projection layer should be created to match dimensions.
    """
    # Input (32) != Units (64)
    layer = ResidualBlock(units=base_config["output_dim"])
    
    y = layer(random_input)
    
    # Check output shape
    assert y.shape == (base_config["batch_size"], base_config["output_dim"])
    
    # Verify projection WAS created
    assert layer.projection is not None, "Projection should exist when dimensions differ."
    assert isinstance(layer.projection, keras.layers.Dense)

def test_training_argument(base_config, random_input):
    """Test that the layer accepts the 'training' argument."""
    layer = ResidualBlock(units=base_config["output_dim"])
    
    # Ensure it runs without error for both True and False
    out_train = layer(random_input, training=True)
    out_infer = layer(random_input, training=False)
    
    # Shapes should match
    assert out_train.shape == out_infer.shape

def test_serialization(base_config, random_input):
    """Test that the layer can be serialized and deserialized."""
    layer = ResidualBlock(units=base_config["output_dim"], dropout_rate=0.5)
    
    # Run once to build weights
    layer(random_input)
    
    # Serialize
    serialized_config = keras.saving.serialize_keras_object(layer)
    
    # Deserialize
    new_layer = keras.saving.deserialize_keras_object(serialized_config)
    
    assert isinstance(new_layer, ResidualBlock)
    assert new_layer.units == base_config["output_dim"]
    assert new_layer.dropout_rate == 0.5

def test_model_save_and_load(base_config, tmp_path):
    """
    Test full model saving/loading with this custom layer.
    Uses pytest's `tmp_path` fixture for automatic cleanup.
    """
    # Create a simple Functional API model
    inputs = keras.Input(shape=(base_config["input_dim"],))
    x = ResidualBlock(units=base_config["output_dim"])(inputs)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    
    # Define save path using tmp_path fixture
    save_path = tmp_path / "test_model.keras"
    
    # Save
    model.save(save_path)
    
    # Load back
    loaded_model = keras.models.load_model(save_path)
    
    # 1. Check if the layer type is preserved
    # The second layer (index 1) should be our ResidualBlock
    assert isinstance(loaded_model.layers[1], ResidualBlock)
    
    # 2. Check basic inference equality
    x_test = np.random.random((1, base_config["input_dim"])).astype("float32")
    
    original_pred = model.predict(x_test, verbose=0)
    loaded_pred = loaded_model.predict(x_test, verbose=0)
    
    # Use numpy testing for array comparison with tolerance
    np.testing.assert_allclose(original_pred, loaded_pred, atol=1e-5)