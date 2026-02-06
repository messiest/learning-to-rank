"""Unit tests for Keras Tuner hypermodels."""

import pytest
import keras
import keras_tuner as kt
from learning_to_rank.tuning.hypermodels import (
    ListNetHyperModel, 
    ResListNetHyperModel, 
    TransformerRankerHyperModel
)

@pytest.mark.parametrize("hypermodel_cls", [
    ListNetHyperModel, 
    ResListNetHyperModel, 
    TransformerRankerHyperModel
])
def test_hypermodel_build(hypermodel_cls):
    """Verify that each hypermodel builds and compiles successfully."""
    hypermodel = hypermodel_cls(num_features=136)
    hp = kt.HyperParameters()
    model = hypermodel.build(hp)
    
    assert isinstance(model, keras.Model)
    
    # Keras 3 way to check compiled metrics
    config = model.get_compile_config()
    # config['metrics'] will contain the list of metrics passed to compile
    # We look through the list (which might contain dicts or strings)
    metrics_config = str(config.get('metrics', [])).lower()
    assert "ndcg" in metrics_config

def test_listnet_tapering_logic():
    """Verify the hidden units tapering calculation inside build."""
    hypermodel = ListNetHyperModel()
    hp = kt.HyperParameters()
    hp.values['num_layers'] = 3
    hp.values['base_units'] = 64
    
    model = hypermodel.build(hp)
    
    # Navigate the Functional wrapper to find the internal ListNet instance
    # Layer 0: Input, Layer 1: The actual ListNet model
    internal_model = model.layers[1]
    assert internal_model.hidden_units == [64, 32, 16]