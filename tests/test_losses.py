import pytest
import tensorflow as tf
import numpy as np
from learning_to_rank.losses import ListwiseSoftmaxLoss

def test_loss_padding_invariance():
    """Ensure that padded values do not affect the loss calculation."""
    loss_fn = ListwiseSoftmaxLoss(padding_value=-1.0)
    
    y_true = tf.constant([[2.0, 1.0, -1.0]], dtype=tf.float32)
    y_pred = tf.constant([[10.0, 0.0, 5.0]], dtype=tf.float32)
    
    # Loss with padding
    loss_with_padding = loss_fn(y_true, y_pred)
    
    # Loss without padding (only the first two elements)
    y_true_no_pad = tf.constant([[2.0, 1.0]], dtype=tf.float32)
    y_pred_no_pad = tf.constant([[10.0, 0.0]], dtype=tf.float32)
    loss_no_padding = loss_fn(y_true_no_pad, y_pred_no_pad)
    
    np.testing.assert_allclose(loss_with_padding, loss_no_padding, atol=1e-5)

def test_loss_directionality():
    """Test that better rankings result in lower loss."""
    loss_fn = ListwiseSoftmaxLoss()
    
    y_true = tf.constant([[3.0, 0.0]], dtype=tf.float32)
    
    # Good prediction (high score for high relevance)
    y_pred_good = tf.constant([[10.0, -10.0]], dtype=tf.float32)
    # Bad prediction (low score for high relevance)
    y_pred_bad = tf.constant([[-10.0, 10.0]], dtype=tf.float32)
    
    loss_good = loss_fn(y_true, y_pred_good)
    loss_bad = loss_fn(y_true, y_pred_bad)
    
    assert loss_good < loss_bad

def test_serialization():
    """Test that the loss can be serialized and deserialized."""
    loss_fn = ListwiseSoftmaxLoss(temperature=2.5, padding_value=-99.0)
    config = loss_fn.get_config()
    
    new_loss_fn = ListwiseSoftmaxLoss.from_config(config)
    assert new_loss_fn.temperature == 2.5
    assert new_loss_fn.padding_value == -99.0