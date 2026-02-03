import pytest
import keras
import numpy as np
import tensorflow as tf
from learning_to_rank.callbacks import TransformerWarmupCallback

def test_warmup_callback_serialization():
    """Verify that the callback can survive a Keras serialization cycle."""
    orig_cb = TransformerWarmupCallback(warmup_epochs=7)
    config = keras.saving.serialize_keras_object(orig_cb)
    new_cb = keras.saving.deserialize_keras_object(config)
    
    assert new_cb.warmup_epochs == 7
    assert isinstance(new_cb, TransformerWarmupCallback)

def test_warmup_callback_execution():
    """Verify that the learning rate is correctly scaled during warmup."""
    # 1. Setup a tiny toy model
    model = keras.Sequential([keras.layers.Dense(1, input_shape=(5,))])
    initial_lr = 0.01
    optimizer = keras.optimizers.Adam(learning_rate=initial_lr)
    model.compile(optimizer=optimizer, loss="mse")

    # 2. Initialize callback with 2 warmup epochs
    warmup_epochs = 2
    callback = TransformerWarmupCallback(warmup_epochs=warmup_epochs)
    callback.set_model(model)

    # 3. Simulate Training Begin (Triggers on_train_begin)
    # This should set LR to 10% of initial
    callback.on_train_begin()
    current_lr = float(keras.ops.convert_to_numpy(model.optimizer.learning_rate))
    assert current_lr == pytest.approx(initial_lr * 0.1)

    # 4. Simulate Epoch 0 (First warmup step)
    # new_lr = initial_lr * (0 + 1) / 2 = 0.005
    callback.on_epoch_begin(0)
    current_lr = float(keras.ops.convert_to_numpy(model.optimizer.learning_rate))
    assert current_lr == pytest.approx(0.005)

    # 5. Simulate Epoch 1 (Final warmup step)
    # new_lr = initial_lr * (1 + 1) / 2 = 0.01
    callback.on_epoch_begin(1)
    current_lr = float(keras.ops.convert_to_numpy(model.optimizer.learning_rate))
    assert current_lr == pytest.approx(initial_lr)

    # 6. Simulate Epoch 2 (Post-warmup, should not change)
    callback.on_epoch_begin(2)
    current_lr = float(keras.ops.convert_to_numpy(model.optimizer.learning_rate))
    assert current_lr == pytest.approx(initial_lr)