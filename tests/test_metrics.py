import pytest
import tensorflow as tf
import numpy as np
from learning_to_rank.metrics import NDCGMetric

def test_ndcg_perfect_ranking():
    """If predictions match the label order, NDCG should be 1.0."""
    metric = NDCGMetric(top_k=2)
    y_true = tf.constant([[3.0, 1.0, 0.0]], dtype=tf.float32)
    y_pred = tf.constant([[10.0, 5.0, 0.0]], dtype=tf.float32)
    
    metric.update_state(y_true, y_pred)
    # Both DCG and IDCG use labels [3, 1]
    assert metric.result().numpy() == pytest.approx(1.0)

def test_ndcg_at_k():
    """Test that only top K items are considered."""
    metric = NDCGMetric(top_k=1)
    # Preds rank second item first. 
    # At K=1, it sees label 1.0. IDCG at K=1 sees label 3.0.
    y_true = tf.constant([[3.0, 1.0]], dtype=tf.float32)
    y_pred = tf.constant([[0.0, 10.0]], dtype=tf.float32)
    
    metric.update_state(y_true, y_pred)
    
    expected_dcg = (2**1.0 - 1) / (np.log2(1 + 1)) # = 1.0
    expected_idcg = (2**3.0 - 1) / (np.log2(1 + 1)) # = 7.0
    assert metric.result().numpy() == pytest.approx(1.0 / 7.0)

def test_ndcg_padding():
    """Ensure padding doesn't boost the IDCG or DCG."""
    metric = NDCGMetric(top_k=5)
    y_true = tf.constant([[3.0, -1.0, -1.0]], dtype=tf.float32)
    y_pred = tf.constant([[10.0, 100.0, 100.0]], dtype=tf.float32)
    
    metric.update_state(y_true, y_pred)
    # The high scores for padding should be ignored. IDCG and DCG should be equal.
    assert metric.result().numpy() == pytest.approx(1.0)