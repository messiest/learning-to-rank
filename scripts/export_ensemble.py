import tensorflow as tf
import os
import shutil
import argparse
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from learning_to_rank.serving_model.ensemble import EnsembleModel

def main(args):
    # Ensure export path directory structure exists (e.g., ensemble/1/)
    if os.path.exists(args.export_path):
        print(f"Cleaning existing export at {args.export_path}")
        shutil.rmtree(args.export_path)
    os.makedirs(os.path.dirname(args.export_path), exist_ok=True)

    # Instantiate Ensemble (handles .keras and SavedModel)
    ensemble = EnsembleModel(args.model_paths)
    
    print(f"Exporting Ensemble to SavedModel: {args.export_path}")
    tf.saved_model.save(
        ensemble, 
        args.export_path,
        signatures={'serving_default': ensemble.__call__}
    )
    print("✅ Export Success. Run 'docker-compose up' to serve.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_paths", nargs="+", help="Paths to .keras or SavedModel files")
    parser.add_argument("--export_path", default="../serving_model/ensemble/1")
    args = parser.parse_args()
    main(args)
