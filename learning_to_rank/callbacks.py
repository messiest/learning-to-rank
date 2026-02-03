"""Custom Keras callbacks for Learning to Rank training."""

from __future__ import annotations
import keras
import tensorflow as tf

__all__ = ["TransformerWarmupCallback"]

@keras.saving.register_keras_serializable(package="LTRCustomCallbacks")
class TransformerWarmupCallback(keras.callbacks.Callback):
    """Linearly increases learning rate from a small value to the target LR.
    
    This helps stabilize Transformers during the initial phase of training.
    """
    def __init__(self, warmup_epochs: int = 5, **kwargs):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.initial_lr = None

    def on_train_begin(self, logs=None):
        # Capture the LR set by the tuner/optimizer at start
        self.initial_lr = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
        # Start at 10% of the target LR
        keras.backend.set_value(self.model.optimizer.learning_rate, self.initial_lr * 0.1)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            # Linear increase calculation
            new_lr = self.initial_lr * ((epoch + 1) / self.warmup_epochs)
            keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
            # Log the change
            tf.print(f"\n[WARMUP] Epoch {epoch+1}: Setting LR to {new_lr:.6f}")

    def get_config(self):
        return {
            "warmup_epochs": self.warmup_epochs,
        }