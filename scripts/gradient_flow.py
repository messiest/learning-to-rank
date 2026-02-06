import numpy as np
import tensorflow as tf
import keras

from learning_to_rank.models.listnet import ResListNet

def verify_gradient_flow(model, batch_size=32, num_docs=100, num_features=136):
    """
    Simulates a training step to verify that gradients flow from 
    the output back to the input weights.
    """
    print("--- Gradient Flow Verification ---")
    
    # 1. Create dummy input and labels
    x = tf.random.normal((batch_size, num_docs, num_features))
    # Labels for ListNet are typically relevance scores [Batch, Docs]
    y_true = tf.cast(tf.random.uniform((batch_size, num_docs), maxval=5, dtype=tf.int32), tf.float32)

    # 2. Open a GradientTape to monitor the forward pass
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    
    with tf.GradientTape() as tape:
        # Watch the input explicitly to check sensitivity
        tape.watch(x)
        
        # Forward pass
        y_pred = model(x, training=True)
        
        # Compute a simple MSE loss for the mock step
        loss = keras.losses.mean_squared_error(y_true, y_pred)
        loss = tf.reduce_mean(loss)

    # 3. Calculate gradients with respect to ALL trainable weights
    grads = tape.gradient(loss, model.trainable_variables)

    # 4. Analyze the gradients
    print(f"Total trainable weight tensors: {len(model.trainable_variables)}")
    
    none_grads = [model.trainable_variables[i].name for i, g in enumerate(grads) if g is None]
    zero_grads = [model.trainable_variables[i].name for i, g in enumerate(grads) if g is not None and tf.reduce_sum(tf.abs(g)) == 0]

    if none_grads:
        print(f"❌ WARNING: Gradients are NONE for: {none_grads}")
    if zero_grads:
        print(f"⚠️ WARNING: Gradients are ZERO (Vanished) for: {zero_grads}")
    
    if not none_grads and not zero_grads:
        # Check the magnitude of the first layer vs last layer
        first_layer_grad_norm = tf.norm(grads[0]).numpy()
        last_layer_grad_norm = tf.norm(grads[-1]).numpy()
        
        print(f"✅ Gradient flow confirmed.")
        print(f"   - First layer gradient norm: {first_layer_grad_norm:.6f}")
        print(f"   - Last layer gradient norm:  {last_layer_grad_norm:.6f}")
        print(f"   - Ratio (First/Last): {first_layer_grad_norm/last_layer_grad_norm:.4f}")

# Execute
model = ResListNet()
verify_gradient_flow(model)