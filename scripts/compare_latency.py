import os
import time

import tensorflow as tf
import numpy as np
from scipy.stats import pearsonr


# --- Configuration ---
TEACHER_PATH = "../serving_model/ensemble/1"        # Path to Ensemble
STUDENT_PATH = "../models/distilled_student"        # Path to Student
INPUT_SHAPE = (136,)                                # MSLR Feature count
NUM_SAMPLES = 5000                                  # How many items to rank
BATCH_SIZE = 1                                      # 1 = Real-time/Online scoring simulation

def get_random_data(num_samples):
    """Generates random data to simulate ranking features."""
    return np.random.rand(num_samples, INPUT_SHAPE[0]).astype(np.float32)

def benchmark_model(model_path, data, model_name="Model"):
    print(f"\n--- Benchmarking: {model_name} ---")
    
    # 1. Load Model
    start_load = time.time()
    model = tf.saved_model.load(model_path)
    infer = model.signatures['serving_default']
    print(f"Load Time: {time.time() - start_load:.4f}s")

    # 2. Identify Output Key (Dynamic)
    key = list(infer.structured_outputs.keys())[0]

    # 3. Warm-up (Critical for fair timing)
    # TensorFlow graphs optimize on the first few runs. We don't want to count this.
    print("Warming up...")
    dummy_data = tf.constant(data[:10])
    _ = infer(dummy_data)

    # 4. Run Benchmark
    print(f"Running inference on {len(data)} samples (Batch Size: {BATCH_SIZE})...")
    
    # Create batches
    dataset = tf.data.Dataset.from_tensor_slices(data).batch(BATCH_SIZE)
    
    predictions = []
    start_time = time.time()
    
    for batch in dataset:
        result = infer(batch)
        score = result[key].numpy()
        predictions.append(score)
        
    end_time = time.time()
    
    # 5. Calculate Stats
    total_time = end_time - start_time
    avg_latency_ms = (total_time / len(data)) * 1000  # Milliseconds
    throughput = len(data) / total_time
    
    print(f"Total Time: {total_time:.4f}s")
    print(f"Average Latency: {avg_latency_ms:.4f} ms/sample")
    print(f"Throughput: {throughput:.2f} samples/sec")
    
    return np.concatenate(predictions, axis=0), avg_latency_ms

def main():
    # Generate Synthetic Data (Faster than loading text files for pure speed tests)
    print("Generating synthetic test data...")
    X_test = get_random_data(NUM_SAMPLES)

    # --- Benchmark Teacher ---
    teacher_preds, teacher_latency = benchmark_model(TEACHER_PATH, X_test, "Teacher (Ensemble)")

    # --- Benchmark Student ---
    student_preds, student_latency = benchmark_model(STUDENT_PATH, X_test, "Student (Distilled)")

    # --- Comparisons ---
    print("\n" + "="*30)
    print("       FINAL RESULTS       ")
    print("="*30)
    
    # 1. Speedup Factor
    speedup = teacher_latency / student_latency
    print(f"Speedup: {speedup:.2f}x FASTER")
    print(f"(Student is {speedup:.2f} times faster than Ensemble)")
    
    # 2. Fidelity Check (Pearson Correlation)
    # How well does the student mimic the teacher?
    # 1.0 = Perfect match, 0.0 = No correlation
    corr, _ = pearsonr(teacher_preds.flatten(), student_preds.flatten())
    print(f"Fidelity (Pearson Correlation): {corr:.4f}")
    
    if corr > 0.9:
        print("✅ Excellent Distillation! Student behaves almost exactly like Teacher.")
    elif corr > 0.75:
        print("⚠️ Good Distillation. Some nuance lost, but ranking order is likely preserved.")
    else:
        print("❌ Poor Distillation. Student did not learn the Teacher's patterns well.")

if __name__ == "__main__":
    main()