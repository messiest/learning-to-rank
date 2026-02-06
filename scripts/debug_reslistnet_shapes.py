import numpy as np
import tensorflow as tf

from learning_to_rank.models.listnet import ResListNet

def debug_reslistnet_shapes(batch_size=32, num_docs=100, num_features=136):
    """Simulates a forward pass and prints internal shapes for debugging."""
    print("--- ResListNet Shape Verification ---")
    
    # Initialize model
    model = ResListNet()
    
    # Create dummy data
    dummy_input = tf.random.normal((batch_size, num_docs, num_features))
    print(f"Input Shape: {dummy_input.shape}")

    # Trigger 'build' and 'call' manually
    # We call it once to initialize everything
    outputs = model(dummy_input, training=False)
    
    print(f"\nModel Summary (verified build):")
    model.summary()

    print(f"\nInternal Block Trace:")
    x = dummy_input
    for i, block in enumerate(model.res_blocks):
        residual = x
        block_out = block(x)
        
        # Check for projection
        if block.projection is not None:
            proj_residual = block.projection(residual)
            print(f"Block {i}: Input {residual.shape} -> Proj {proj_residual.shape} | Main {block_out.shape}")
            assert proj_residual.shape == block_out.shape, f"Shape mismatch in Block {i}!"
        else:
            print(f"Block {i}: Input {residual.shape} -> Identity | Main {block_out.shape}")
            assert residual.shape == block_out.shape, f"Identity mismatch in Block {i}!"
        
        x = tf.keras.layers.Add()([block_out, proj_residual if block.projection else residual])
        x = tf.keras.layers.Activation("relu")(x)
        print(f"      -> Post-Addition Shape: {x.shape}")

    print(f"\nFinal Score Shape: {outputs.shape}")
    print("-------------------------------------")
    print("[SUCCESS] All residual connections are mathematically aligned.")

# Execute the debug
debug_reslistnet_shapes()