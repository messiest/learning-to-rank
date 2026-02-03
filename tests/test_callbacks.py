from learning_to_rank.callbacks import TransformerWarmupCallback
import keras

def test_callback_serialization():
    """Ensures the warmup callback can be saved and reloaded by Keras."""
    original_callback = TransformerWarmupCallback(warmup_epochs=7)
    config = keras.saving.serialize_keras_object(original_callback)
    
    revived_callback = keras.saving.deserialize_keras_object(config)
    assert revived_callback.warmup_epochs == 7