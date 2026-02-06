import os
import argparse
import tensorflow as tf

from learning_to_rank.data import combine_tfrecords


def main():
    parser = argparse.ArgumentParser(description="Merge multiple TFRecord files into one.")
    
    # Input files argument (supports wildcard expansion by the shell)
    parser.add_argument(
        "input_files", 
        nargs="+", 
        help="List of input .tfrecord files (e.g., data/part-*.tfrecord)"
    )
    
    # Output file argument
    parser.add_argument(
        "--output", "-o", 
        required=True, 
        help="Path to the destination combined .tfrecord file"
    )
    
    # Optional compression
    parser.add_argument(
        "--compression", "-c", 
        type=str, 
        default=None, 
        choices=['GZIP', 'ZLIB', 'NONE'],
        help="Compression type of input/output files (default: None)"
    )

    args = parser.parse_args()

    # Handle shell globbing explicitly if the OS didn't do it (mostly for Windows cmd)
    # or if the user passed a quoted string like "*.tfrecord"
    resolved_files = []
    for path in args.input_files:
        if "*" in path or "?" in path:
            resolved_files.extend(tf.io.gfile.glob(path))
        else:
            resolved_files.append(path)
            
    # Remove duplicates and sort for deterministic order
    resolved_files = sorted(list(set(resolved_files)))

    combine_tfrecords(
        input_files=resolved_files, 
        output_path=args.output, 
        compression_type=None if args.compression == 'NONE' else args.compression
    )

if __name__ == "__main__":
    main()
