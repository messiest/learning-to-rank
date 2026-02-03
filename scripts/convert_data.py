import os
import argparse
import sys

# Ensure we can import from the project root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from learning_to_rank.data import convert_libsvm_to_tfrecord
from learning_to_rank.config import NUM_FEATURES

def main():
    parser = argparse.ArgumentParser(description="Convert MSLR Text Data to TFRecords")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset folder in data/raw (e.g., MSLR-WEB10K)")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Base raw data directory")
    parser.add_argument("--processed_dir", type=str, default="data/processed", help="Base processed data directory")
    parser.add_argument("--force", action="store_true", help="Overwrite existing TFRecord files")
    args = parser.parse_args()

    # Paths
    raw_dataset_path = os.path.join(args.raw_dir, args.dataset_name)
    processed_dataset_path = os.path.join(args.processed_dir, args.dataset_name)

    # 1. Validation
    if not os.path.exists(raw_dataset_path):
        print(f" [ERROR] Could not find dataset at {raw_dataset_path}")
        print(f"         Did you run 'python scripts/download_data.py --dataset {args.dataset_name}'?")
        sys.exit(1)

    print(f" [INFO] Converting dataset: {args.dataset_name}")
    print(f"        Input:  {raw_dataset_path}")
    print(f"        Output: {processed_dataset_path}")

    # 2. Iterate over Folds (Fold1 -> Fold5)
    folds = [f"Fold{i}" for i in range(1, 6)]
    splits = ["train", "vali", "test"]
    
    # MSLR convention: train.txt, vali.txt, test.txt
    # Sometimes they are named differently, but for MSLR-WEB10K/30K this is standard.
    
    total_files = 0
    
    for fold in folds:
        raw_fold_dir = os.path.join(raw_dataset_path, fold)
        processed_fold_dir = os.path.join(processed_dataset_path, fold)
        
        if not os.path.exists(raw_fold_dir):
            print(f" [WARN] Skipping {fold} (directory not found)")
            continue
            
        # Create output directory for this fold
        os.makedirs(processed_fold_dir, exist_ok=True)
        
        for split in splits:
            txt_filename = f"{split}.txt"
            tfrecord_filename = f"{split}.tfrecord"
            
            input_path = os.path.join(raw_fold_dir, txt_filename)
            output_path = os.path.join(processed_fold_dir, tfrecord_filename)
            
            if os.path.exists(input_path):
                print(f"\nProcessing {fold}/{split}...")
                convert_libsvm_to_tfrecord(
                    input_files=[input_path],
                    output_path=output_path,
                    num_features=NUM_FEATURES,
                    force_overwrite=args.force
                )
                total_files += 1
            else:
                # Some datasets might use 'val.txt' instead of 'vali.txt'
                # Simple fallback check
                if split == "vali":
                    fallback_path = os.path.join(raw_fold_dir, "val.txt")
                    if os.path.exists(fallback_path):
                        print(f"\nProcessing {fold}/{split} (found as val.txt)...")
                        convert_libsvm_to_tfrecord(
                            input_files=[fallback_path],
                            output_path=output_path,
                            num_features=NUM_FEATURES,
                            force_overwrite=args.force
                        )
                        total_files += 1

    print(f"\n [SUCCESS] Pipeline complete. Converted {total_files} files.")

if __name__ == "__main__":
    main()
