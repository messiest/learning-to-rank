import pytest
import numpy as np
import tensorflow as tf
import os
from learning_to_rank import data
from learning_to_rank.data import parse_libsvm_line, serialize_sequence_example, parse_tfrecord_fn, combine_tfrecords, stream_file_list_generator, convert_libsvm_to_tfrecord, truncate_fn, build_dataset

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

# ==============================================================================
# 1. GENERATOR & PARSING LOGIC
# ==============================================================================

def test_stream_generator_grouping(tmp_path):
    """Test that the generator correctly groups documents by QID."""
    # Create a dummy file with 2 queries:
    # Query 1: 2 docs
    # Query 2: 1 doc
    content = (
        "2 qid:1 1:0.5\n"
        "1 qid:1 1:0.3\n"
        "3 qid:2 1:0.9\n"
    )
    f = tmp_path / "test.txt"
    f.write_text(content, encoding='utf-8')
    
    gen = stream_file_list_generator([str(f)], num_features=1)
    results = list(gen)
    
    # Expect 2 groups
    assert len(results) == 2
    
    # Check Query 1
    feats1, labels1 = results[0]
    assert len(labels1) == 2
    assert labels1[0] == 2.0
    assert labels1[1] == 1.0
    
    # Check Query 2
    feats2, labels2 = results[1]
    assert len(labels2) == 1
    assert labels2[0] == 3.0

def test_stream_generator_missing_file(tmp_path):
    """Test that the generator gracefully skips missing files."""
    # Pass a non-existent path
    gen = stream_file_list_generator([str(tmp_path / "ghost.txt")], num_features=5)
    results = list(gen)
    assert len(results) == 0

# ==============================================================================
# 2. CONVERSION & OVERWRITE LOGIC
# ==============================================================================

def test_convert_overwrite_protection(tmp_path):
    """Test that conversion respects the force_overwrite flag."""
    input_file = tmp_path / "in.txt"
    input_file.write_text("1 qid:1 1:0.5", encoding='utf-8')
    
    output_file = tmp_path / "out.tfrecord"
    output_file.touch() # Create empty file
    
    # 1. Run with force_overwrite=False (Default)
    # The function should return early and NOT write anything
    convert_libsvm_to_tfrecord([str(input_file)], str(output_file), num_features=1, force_overwrite=False)
    assert output_file.stat().st_size == 0 # Should still be empty
    
    # 2. Run with force_overwrite=True
    convert_libsvm_to_tfrecord([str(input_file)], str(output_file), num_features=1, force_overwrite=True)
    assert output_file.stat().st_size > 0 # Should now contain data

# ==============================================================================
# 3. PIPELINE UTILITIES (Truncation)
# ==============================================================================

def test_truncate_fn():
    """Unit test for the truncation helper function."""
    # Create fake batch of 5 documents, 2 features
    feats = tf.ones((5, 2))
    labels = tf.ones((5,))
    
    # Truncate to 3
    t_feats, t_labels = truncate_fn(feats, labels, max_list_size=3)
    
    assert t_feats.shape == (3, 2)
    assert t_labels.shape == (3,)

def test_dataset_truncation_integration(tmp_path, monkeypatch):
    """Integration test: Verify build_dataset actually applies truncation."""
    monkeypatch.setattr(data, "NUM_FEATURES", 1)
    
    # Create 1 query with 10 documents
    features = np.zeros((10, 1), dtype=np.float32)
    labels = np.zeros((10,), dtype=np.float32)
    serialized = serialize_sequence_example(features, labels)
    
    tfrecord_path = tmp_path / "trunc.tfrecord"
    with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
        writer.write(serialized)
        
    # Build dataset with max_list_size=4
    ds = build_dataset(str(tfrecord_path), batch_size=1, max_list_size=4, shuffle=False)
    
    for x, y in ds:
        # Shape should be [Batch=1, List=4, Feat=1]
        assert x.shape == (1, 4, 1)
        assert y.shape == (1, 4)

# ==============================================================================
# 4. TFRECORD COMBINATION
# ==============================================================================

def test_combine_tfrecords(tmp_path):
    """Test merging multiple TFRecord files."""
    # Create 2 source files
    file1 = tmp_path / "part1.tfrecord"
    file2 = tmp_path / "part2.tfrecord"
    combined = tmp_path / "combined.tfrecord"
    
    # Write 1 record to File 1
    with tf.io.TFRecordWriter(str(file1)) as w:
        w.write(b"record1")
        
    # Write 2 records to File 2
    with tf.io.TFRecordWriter(str(file2)) as w:
        w.write(b"record2")
        w.write(b"record3")
        
    # Run Combine
    combine_tfrecords([str(file1), str(file2)], str(combined))
    
    # Verify contents
    dataset = tf.data.TFRecordDataset(str(combined))
    records = list(dataset.as_numpy_iterator())
    
    assert len(records) == 3
    assert b"record1" in records
    assert b"record2" in records
    assert b"record3" in records

def test_combine_tfrecords_empty_input():
    """Test that combine raises error on empty input list."""
    with pytest.raises(ValueError, match="No input files"):
        combine_tfrecords([], "dummy_output")