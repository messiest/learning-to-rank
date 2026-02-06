"""Data processing and pipeline utilities for Learning to Rank.

This module provides functions for parsing LIBSVM-formatted datasets (like MSLR),
converting them to optimized TFRecord formats using SequenceExamples, and
building high-performance tf.data pipelines for training and inference.
"""

from __future__ import annotations

import os
from typing import Generator, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

# Try to import config, but provide defaults if running standalone
try:
    from .config import BATCH_SIZE, NUM_FEATURES, PADDING_LABEL, MAX_LIST_SIZE
except ImportError:
    NUM_FEATURES = 136
    PADDING_LABEL = -1.0
    BATCH_SIZE = 32
    MAX_LIST_SIZE = 100

__all__ = [
    "parse_libsvm_line",
    "stream_file_list_generator",
    "serialize_sequence_example",
    "convert_libsvm_to_tfrecord",
    "parse_tfrecord_fn",
    "build_dataset",
    "combine_tfrecords"
]

# ==============================================================================
# 1. RAW TEXT PARSING
# ==============================================================================

def parse_libsvm_line(
    line: str, 
    num_features: int
) -> Optional[Tuple[float, str, np.ndarray]]:
    """Parses a single line from an MSLR-style LIBSVM text file.

    Format: <label> qid:<query_id> <feature_index>:<feature_value> ... # <comment>

    Args:
        line: A single string line from the dataset file.
        num_features: The expected number of features (e.g., 136 for MSLR-WEB10K).

    Returns:
        A tuple of (label, query_id, feature_vector) if parsing is successful,
        otherwise None if the line is empty or malformed.
    """
    clean_line = line.split('#')[0].strip()
    if not clean_line:
        return None

    tokens = clean_line.split()
    label = float(tokens[0])
    qid = tokens[1].split(':')[1]
    
    feature_vec = np.zeros(num_features, dtype=np.float32)
    
    for token in tokens[2:]:
        if ':' not in token:
            continue
        feat_idx_str, feat_val_str = token.split(':')
        feat_idx = int(feat_idx_str) - 1  # 1-indexed to 0-indexed
        
        if 0 <= feat_idx < num_features:
            feature_vec[feat_idx] = float(feat_val_str)
            
    return label, qid, feature_vec


def stream_file_list_generator(
    file_list: List[str], 
    num_features: int = NUM_FEATURES
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Generator that yields document features and labels grouped by query ID.

    Args:
        file_list: List of paths to the raw text files.
        num_features: Number of features per document.

    Yields:
        A tuple of (features, labels) where:
            features: np.ndarray of shape [num_docs_in_query, num_features]
            labels: np.ndarray of shape [num_docs_in_query]
    """
    current_qid = None
    current_features = []
    current_labels = []

    for file_path in file_list:
        if not os.path.exists(file_path):
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parsed = parse_libsvm_line(line, num_features)
                if parsed is None:
                    continue
                
                label, qid, feat_vec = parsed

                if qid != current_qid:
                    if current_qid is not None and current_labels:
                        yield np.array(current_features), np.array(current_labels)
                    
                    current_qid = qid
                    current_features = []
                    current_labels = []
                
                current_features.append(feat_vec)
                current_labels.append(label)

    if current_qid is not None and current_labels:
        yield np.array(current_features), np.array(current_labels)


# ==============================================================================
# 2. TFRECORD SERIALIZATION
# ==============================================================================

def serialize_sequence_example(features: np.ndarray, labels: np.ndarray) -> bytes:
    """Serializes a query group into a tf.train.SequenceExample proto.

    Args:
        features: Numpy array of shape [num_docs, num_features].
        labels: Numpy array of shape [num_docs].

    Returns:
        A serialized byte string representing the SequenceExample.
    """
    doc_features = [
        tf.train.Feature(float_list=tf.train.FloatList(value=f)) 
        for f in features
    ]
    
    label_features = [
        tf.train.Feature(float_list=tf.train.FloatList(value=[l])) 
        for l in labels
    ]

    seq_example = tf.train.SequenceExample(
        context=tf.train.Features(feature={
            'list_size': tf.train.Feature(int64_list=tf.train.Int64List(value=[len(labels)]))
        }),
        feature_lists=tf.train.FeatureLists(feature_list={
            'features': tf.train.FeatureList(feature=doc_features),
            'labels': tf.train.FeatureList(feature=label_features)
        })
    )
    return seq_example.SerializeToString()


def convert_libsvm_to_tfrecord(
    input_files: List[str], 
    output_path: str, 
    num_features: int = NUM_FEATURES, 
    force_overwrite: bool = False
) -> None:
    """Converts raw LIBSVM text files into a single TFRecord file.

    Args:
        input_files: List of paths to input text files.
        output_path: Path where the TFRecord file should be saved.
        num_features: Number of features in the input data.
        force_overwrite: If True, will overwrite existing TFRecord files.
    """
    if os.path.exists(output_path) and not force_overwrite:
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    count = 0
    with tf.io.TFRecordWriter(output_path) as writer:
        for features, labels in stream_file_list_generator(input_files, num_features):
            example = serialize_sequence_example(features, labels)
            writer.write(example)
            count += 1
    
    print(f"Successfully wrote {count} queries to {output_path}")


# ==============================================================================
# 3. DATA PIPELINE
# ==============================================================================

def parse_tfrecord_fn(example_proto: bytes) -> Tuple[tf.Tensor, tf.Tensor]:
    """Parses a serialized SequenceExample into feature and label tensors.

    Args:
        example_proto: A serialized SequenceExample protocol buffer.

    Returns:
        A tuple of (features, labels) where:
            features: float32 Tensor of shape [num_docs, num_features]
            labels: float32 Tensor of shape [num_docs]
    """
    context_desc = {'list_size': tf.io.FixedLenFeature([], tf.int64)}
    sequence_desc = {
        'features': tf.io.FixedLenSequenceFeature([NUM_FEATURES], tf.float32),
        'labels': tf.io.FixedLenSequenceFeature([1], tf.float32),
    }
    
    context, sequence = tf.io.parse_single_sequence_example(
        example_proto,
        context_features=context_desc,
        sequence_features=sequence_desc
    )
    
    return sequence['features'], tf.squeeze(sequence['labels'], axis=-1)


def truncate_fn(features: tf.Tensor, labels: tf.Tensor, max_list_size: int):
    """Truncates the feature and label tensors to a maximum list size."""
    # We take the first 'max_list_size' documents for the query
    features = features[:max_list_size, :]
    labels = labels[:max_list_size]
    return features, labels

def build_dataset(
    file_paths: Union[str, List[str]], 
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True,
    cache: bool = True,
    max_list_size: Optional[int] = MAX_LIST_SIZE,
) -> tf.data.Dataset:
    """Creates a tf.data.Dataset optimized for Learning to Rank."""
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.interleave(
        tf.data.TFRecordDataset, 
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not shuffle
    )
    
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    # --- TRUNCATION STRATEGY ---
    if max_list_size:
        # We use a lambda to pass the max_list_size to our truncate function
        dataset = dataset.map(
            lambda f, l: truncate_fn(f, l, max_list_size),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    # --- CACHING STRATEGY ---
    if cache:
        dataset = dataset.cache()
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=([None, NUM_FEATURES], [None]),
        padding_values=(0.0, PADDING_LABEL)
    )
    
    return dataset.prefetch(tf.data.AUTOTUNE)

def combine_tfrecords(input_files, output_path, compression_type=None):
    """
    Combines multiple TFRecord files into a single output file.
    """
    # 1. Validation
    if not input_files:
        raise ValueError("No input files provided to combine.")
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    # 2. Initialize the writer
    # Note: If your source files are GZIP compressed, pass 'GZIP'
    options = tf.io.TFRecordOptions(compression_type=compression_type)
    
    print(f"--- Combining {len(input_files)} files ---")
    print(f"Output: {output_path}")

    total_count = 0
    
    # 3. Stream and Write
    with tf.io.TFRecordWriter(output_path, options=options) as writer:
        # tf.data.TFRecordDataset automatically handles reading from multiple files
        raw_dataset = tf.data.TFRecordDataset(input_files, compression_type=compression_type)
        
        for raw_record in raw_dataset:
            writer.write(raw_record.numpy())
            total_count += 1
            
            if total_count % 5000 == 0:
                print(f"Processed {total_count} records...", end="\r")

    print(f"\n✅ [SUCCESS] Combined {total_count} records into: {output_path}")