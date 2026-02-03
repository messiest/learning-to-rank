"""Custom Keras callbacks for Learning to Rank training."""

from __future__ import annotations
import keras
import tensorflow as tf

__all__ = ["TransformerWarmupCallback"]

@keras.saving.register_keras_serializable(package="LTRCustomCallbacks")
class TransformerWarmupCallback(keras.callbacks.Callback):
    """Linearly increases learning rate from a small value to the target LR.
    
    Stabilizes Transformers during the initial phase of training.
    """
    def __init__(self, warmup_epochs: int = 5, **kwargs):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.initial_lr = None
        self.kwargs = kwargs

    def on_train_begin(self, logs=None):
        """Capture the target learning rate and set the initial small LR."""
        # Keras 3 optimizers store learning_rate as a Variable
        lr_var = self.model.optimizer.learning_rate
        self.initial_lr = float(keras.ops.convert_to_numpy(lr_var))
        
        # Set to 10% of target LR using Keras 3 assignment
        self.model.optimizer.learning_rate = self.initial_lr * 0.1

    def on_epoch_begin(self, epoch, logs=None):
        """Linearly scale the learning rate up to the target during warmup."""
        if self.initial_lr is not None and epoch < self.warmup_epochs:
            # Linear increase calculation
            new_lr = self.initial_lr * ((epoch + 1) / self.warmup_epochs)
            self.model.optimizer.learning_rate = new_lr
            tf.print(f"\n[WARMUP] Epoch {epoch+1}: Setting LR to {new_lr:.6f}")

    def get_config(self) -> dict:
        """Returns the config dictionary for serialization."""
        return {
            "warmup_epochs": self.warmup_epochs,
        }

    @classmethod
    def from_config(cls, config: dict) -> TransformerWarmupCallback:
        """Creates an instance of the callback from a config dictionary."""
        return cls(**config)