import os
import sys
import argparse
import tensorflow as tf
import keras_tuner as kt

# Ensure the parent directory is in the path so we can import our package
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from learning_to_rank.data import build_dataset
from learning_to_rank.tuning.hypermodels import (
    ListNetHyperModel, 
    ResListNetHyperModel, 
    TransformerRankerHyperModel
)

# --- GLOBAL CONFIG ---
FEATURE_COUNT = 136  # Default for MSLR
HYPERMODEL_MAP = {
    "listnet": ListNetHyperModel,
    "reslistnet": ResListNetHyperModel,
    "transformer": TransformerRankerHyperModel
}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Unified LTR Training & Tuning")
    parser.add_argument("--model_type", type=str, default="reslistnet", choices=["listnet", "reslistnet", "transformer"])
    parser.add_argument("--data_dir", type=str, default="./data/processed/MSLR-WEB10K")
    parser.add_argument("--output_dir", type=str, default="tuning_results")
    parser.add_argument("--model_save_dir", type=str, default="./models/tuned")
    parser.add_argument("--single_fold", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true", help="Clear previous tuning logs")

    return parser.parse_args()

def get_fold_config(base_path):
    folds = {}
    for i in range(1, 6):
        fold_name = f"Fold{i}"
        fold_dir = os.path.join(base_path, fold_name)
        folds[fold_name] = {
            "training": os.path.join(fold_dir, "train.tfrecord"),
            "validation": os.path.join(fold_dir, "vali.tfrecord"),
            "testing": os.path.join(fold_dir, "test.tfrecord")
        }
    return folds

def main():
    args = parse_arguments()

    os.makedirs(args.model_save_dir, exist_ok=True)

    # 1. Model Class Selection
    if args.model_type not in HYPERMODEL_MAP:
        raise ValueError(f"Invalid model_type. Choose from: {list(HYPERMODEL_MAP.keys())}")
    
    HyperModelClass = HYPERMODEL_MAP[args.model_type]
    dataset_folds = get_fold_config(args.data_dir)
    
    # Filter folds if single_fold is specified
    target_folds = {args.single_fold: dataset_folds[args.single_fold]} if args.single_fold else dataset_folds

    results = {}

    for fold, paths in target_folds.items():
        print(f"\n{'='*60}\n  Model: {args.model_type.upper()} | Fold: {fold}\n{'='*60}")
        
        # 2. Build Datasets
        train_dataset = build_dataset(paths["training"], batch_size=args.batch_size)
        val_dataset = build_dataset(paths["validation"], batch_size=args.batch_size)
        test_dataset = build_dataset(paths["testing"], batch_size=args.batch_size)

        # 3. Setup Tuner
        # We include model_type in the project name to keep trials separate
        hypermodel = HyperModelClass(num_features=FEATURE_COUNT)
        
        tuner = kt.Hyperband(
            hypermodel,
            objective=kt.Objective("val_ndcg", direction="max"),
            max_epochs=args.max_epochs,
            factor=3,
            directory=args.output_dir,
            project_name=f"{args.model_type}_{fold}",
            overwrite=args.overwrite
        )
        
        stop_early = tf.keras.callbacks.EarlyStopping(
            monitor='val_ndcg', patience=3, mode='max', restore_best_weights=True
        )

        # 4. Search
        print(f"Starting HP search for {args.model_type}...")
        tuner.search(
            train_dataset,
            validation_data=val_dataset,
            callbacks=[stop_early],
            verbose=1
        )

        # 5. Retrieve and Retrain Best Model
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f"\nRetraining best {args.model_type} model...")
        
        best_model = tuner.hypermodel.build(best_hps)
        best_model.fit(
            train_dataset,
            epochs=args.max_epochs * 2, # Train longer for the final pass
            validation_data=val_dataset,
            callbacks=[
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_ndcg', factor=0.5, patience=2),
                tf.keras.callbacks.EarlyStopping(monitor='val_ndcg', patience=5, restore_best_weights=True, mode="max")
            ],
            verbose=1
        )
        
        # 6. Final Evaluation
        print(f"\nEvaluating {args.model_type} on {fold} test set...")
        eval_metrics = best_model.evaluate(test_dataset, return_dict=True)
        results[fold] = eval_metrics
        
        # 7. Save Model
        save_name = f"{args.model_type}_{fold}_best.keras"
        save_path = os.path.join(args.model_save_dir, save_name)
        best_model.save(save_path)
        print(f"Model saved: {save_path}")

    # Summary table
    print(f"\n{'='*30}\nFinal {args.model_type.upper()} Results\n{'='*30}")
    for fold, metrics in results.items():
        print(f"{fold}: NDCG={metrics.get('ndcg', 0.0):.4f}")


if __name__ == "__main__":
    main()