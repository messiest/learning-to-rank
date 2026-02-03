import tensorflow as tf
import os
import sys
import argparse

# Ensure we can import from the project root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from learning_to_rank.losses import ListwiseSoftmaxLoss
from learning_to_rank.metrics import NDCGMetric
from learning_to_rank.models.listnet import ListNet, ResListNet
from learning_to_rank.models.transformer import TransformerRanker
from learning_to_rank.data import build_dataset
from learning_to_rank.config import NUM_FEATURES, PADDING_LABEL

# --- Configuration ---
TRAIN_PATTERN = "data/processed/MSLR-WEB10K/Fold1/train.tfrecord"
VALID_PATTERN = "data/processed/MSLR-WEB10K/Fold1/vali.tfrecord"
MODEL_SAVE_DIR = "models/"

LEARNING_RATE = 0.01

def get_model(model_type, num_features):
    model_type = model_type.lower()
    if model_type == 'listnet':
        return ListNet(num_features=num_features)
    elif model_type == 'reslistnet':
        return ResListNet(num_features=num_features)
    elif model_type == 'transformer':
        return TransformerRanker(num_features=num_features)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def main(args):
    # 1. Resolve File Paths
    # build_dataset expects a list of file paths, so we expand the glob pattern (e.g., "*.tfrecord")
    train_files = tf.io.gfile.glob(args.train_path)
    val_files = tf.io.gfile.glob(args.valid_path)
    
    if not train_files:
        raise ValueError(f"No files found for pattern: {args.train_path}")
    
    print(f"--- Initializing Data Pipeline ---")
    print(f"Found {len(train_files)} training files.")
    print(f"Found {len(val_files)} validation files.")

    # 2. Build Datasets using your library function
    # Note: Your build_dataset function handles shuffling internally.
    train_dataset = build_dataset(train_files, batch_size=args.batch_size)
    val_dataset = build_dataset(val_files, batch_size=args.batch_size)

    # 3. Initialize Model
    print(f"\n--- Initializing Model: {args.model_type} ---")
    model = get_model(args.model_type, NUM_FEATURES)

    # 4. Compile
    # We use MSE. Since build_dataset provides Listwise data (Batch, List_Size, Feats),
    # the Keras models (Dense layers) will automatically apply to every item in the list.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
            clipnorm=1.0
        ),
        loss=ListwiseSoftmaxLoss(padding_value=PADDING_LABEL),
        metrics=[NDCGMetric(top_k=5, padding_value=PADDING_LABEL)]
    )

    # 5. Train
    print(f"Starting training for {args.epochs} epochs...")
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True
    )

    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_ndcg',      # Watch the validation NDCG
        mode='max',              # We want it to go UP
        factor=0.5,              # Cut LR by half when stuck
        patience=2,              # Wait 2 epochs before cutting
        min_lr=1e-6,
        verbose=0,
    )

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=[early_stopping, lr_callback],
        verbose=2
    )

    # 6. Save Model
    save_path = os.path.join(MODEL_SAVE_DIR, f"{args.model_name}.keras")
    print(f"\nSaving {args.model_type} model to: {save_path}")
    model.save(save_path)
    print("✅ Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_type", type=str, required=True, 
                        choices=['listnet', 'reslistnet', 'transformer'],
                        help="Architecture to train")
    parser.add_argument("--model_name", type=str, required=True, 
                        help="Name of the output model folder")
    parser.add_argument("--train_path", type=str, default=TRAIN_PATTERN,
                        help="Glob pattern for Training TFRecords")
    parser.add_argument("--valid_path", type=str, default=VALID_PATTERN,
                        help="Glob pattern for Validation TFRecords")
    
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32) # Lower default batch size for Listwise (32 queries = thousands of docs)
    parser.add_argument("--lr", type=float, default=0.001)
    
    args = parser.parse_args()
    main(args)