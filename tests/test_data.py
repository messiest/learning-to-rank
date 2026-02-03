import pytest
import numpy as np
import tensorflow as tf
import os
from learning_to_rank import data
from learning_to_rank.data import parse_libsvm_line, serialize_sequence_example, parse_tfrecord_fn

def test_parse_libsvm_line():
    """Test that a standard MSLR line is parsed correctly (1-indexing handled)."""
    line = "2 qid:10 1:0.5 3:0.1 # comment"
    num_features = 5
    label, qid, features = parse_libsvm_line(line, num_features)
    
    assert label == pytest.approx(2.0)
    assert qid == "10"
    assert features[0] == pytest.approx(0.5)   # Feature 1 -> index 0
    assert features[2] == pytest.approx(0.1)   # Feature 3 -> index 2
    assert features[1] == pytest.approx(0.0)   # Missing feature -> 0.0
    assert len(features) == 5

def test_parse_libsvm_line_empty():
    """Test handling of empty or comment-only lines."""
    assert parse_libsvm_line("", 5) is None
    assert parse_libsvm_line("# just a comment", 5) is None

def test_serialization_cycle():
    """Test that serializing and parsing returns the original data/shapes."""
    features = np.random.rand(3, 136).astype(np.float32)
    labels = np.array([1.0, 0.0, 2.0], dtype=np.float32)
    
    serialized = serialize_sequence_example(features, labels)
    parsed_features, parsed_labels = parse_tfrecord_fn(serialized)
    
    assert parsed_features.shape == (3, 136)
    assert parsed_labels.shape == (3,)
    np.testing.assert_allclose(features, parsed_features.numpy())
    np.testing.assert_allclose(labels, parsed_labels.numpy())

def test_full_conversion_pipeline(tmp_path, monkeypatch):
    """Test the end-to-end flow from LIBSVM text to TFRecord to Dataset."""
    # Temporarily set NUM_FEATURES to 2 to match our dummy data
    monkeypatch.setattr(data, "NUM_FEATURES", 2)
    
    # 1. Create a dummy LIBSVM file
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    libsvm_file = data_dir / "train.txt"
    # 2 docs, 2 features each = 4 total feature values
    libsvm_file.write_text("3 qid:1 1:1.0 2:0.5\n2 qid:1 1:0.2 2:0.8\n")
    
    # 2. Convert to TFRecord
    tfrecord_path = str(tmp_path / "train.tfrecord")
    from learning_to_rank.data import convert_libsvm_to_tfrecord, build_dataset
    
    convert_libsvm_to_tfrecord([str(libsvm_file)], tfrecord_path, num_features=2)
    
    # 3. Build and consume dataset
    dataset = build_dataset(tfrecord_path, batch_size=1, shuffle=False)
    
    for x, y in dataset.take(1):
        assert x.shape == (1, 2, 2) # [Batch, List, Feat]
        assert y.shape == (1, 2)    # [Batch, List]
        # Check first doc label
        assert y.numpy()[0][0] == 3.0