"""Specialized tuner for the TransformerRanker model."""

import argparse
import sys
import keras
import keras_tuner as kt
import tensorflow as tf

from learning_to_rank.data import build_dataset
from learning_to_rank.metrics import NDCGMetric
from learning_to_rank.tuning.hypermodels import TransformerRankerHyperModel
from learning_to_rank.callbacks import TransformerWarmupCallback

def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Optimize Transformer LTR Model")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--project_name", type=str, default="transformer_tuning")
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    args = parser.parse_args(argv)

    # 1. Specialized HyperModel
    # We increase the search resolution for the Transformer specifically
    hypermodel = TransformerRankerHyperModel(num_features=136)

    # 2. Transformer-Specific Tuner Configuration
    tuner = kt.Hyperband(
        hypermodel,
        objective=kt.Objective("val_ndcg", direction="max"),
        max_epochs=args.max_epochs,
        factor=3,
        directory="models/tuning",
        project_name=args.project_name,
        # Transformers benefit from more executions per trial to reduce noise
        executions_per_trial=2 
    )

    # 3. Optimized Callbacks
    # Learning Rate warm-up is handled here or within the HyperModel lr search
    callbacks = [
        TransformerWarmupCallback(warmup_epochs=args.warmup_epochs),
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        # Reduce LR on plateau is vital for Transformers
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
    ]

    # 4. Data Loading
    train_ds = build_dataset(args.train_path, shuffle=True)
    val_ds = build_dataset(args.val_path, shuffle=False)

    # 5. Execute Search
    print(f" [INFO] Starting Transformer Optimization...")
    tuner.search(
        train_ds,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=2,
    )

    # 6. Save Best Model
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.save("models/best_transformer_optimized.keras")
    print(" [SUCCESS] Optimized Transformer Model Saved.")

if __name__ == "__main__":
    main()
