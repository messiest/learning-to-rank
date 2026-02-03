import tensorflow as tf
import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from learning_to_rank.data import build_dataset
from learning_to_rank.models.listnet import ListNet, ResListNet
from learning_to_rank.models.transformer import TransformerRanker

# Map for selecting the Student architecture
STUDENT_MAP = {
    "listnet": ListNet,
    "reslistnet": ResListNet,
    "transformer": TransformerRanker
}

def distill(args):
    # 1. Load the Teacher (The Ensemble SavedModel)
    print(f"Loading Teacher Ensemble from: {args.teacher_path}")
    teacher = tf.saved_model.load(args.teacher_path)
    teacher_fn = teacher.signatures['serving_default']
    
    # 2. Initialize the Student
    if args.student_type not in STUDENT_MAP:
        raise ValueError(f"Unknown student type. Choose from: {list(STUDENT_MAP.keys())}")
    
    StudentClass = STUDENT_MAP[args.student_type]
    student = StudentClass(num_features=136)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    distill_loss_fn = tf.keras.losses.KLDivergence()
    
    dataset = build_dataset(args.train_records, batch_size=args.batch_size)

    @tf.function
    def train_step(x):
        # Get Teacher predictions
        # Dynamic key lookup to handle varying SavedModel signatures
        out_key = list(teacher_fn.structured_outputs.keys())[0]
        teacher_logits = teacher_fn(x)[out_key]
        
        # Apply Temperature to soften the distribution
        # Higher temperature = more 'information' about secondary rankings
        teacher_probs = tf.nn.softmax(teacher_logits / args.temperature, axis=-1)

        with tf.GradientTape() as tape:
            student_logits = student(x, training=True)
            student_probs = tf.nn.softmax(student_logits / args.temperature, axis=-1)
            
            # KL Divergence helps the student mimic the teacher's 'logic'
            loss = distill_loss_fn(teacher_probs, student_probs)
        
        grads = tape.gradient(loss, student.trainable_variables)
        optimizer.apply_gradients(zip(grads, student.trainable_variables))
        return loss

    print(f"🚀 Distilling Ensemble into {args.student_type} student...")
    for epoch in range(args.epochs):
        epoch_loss = tf.keras.metrics.Mean()
        for x, _ in dataset:
            loss = train_step(x)
            epoch_loss.update_state(loss)
        print(f"Epoch {epoch+1}/{args.epochs} | KLD Loss: {epoch_loss.result():.4f}")

    # 3. Save the Student
    student.save(args.output_path)
    print(f"✅ Distillation Complete. Student saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LTR Knowledge Distillation")
    parser.add_argument("--teacher_path", required=True, help="Path to the Ensemble SavedModel")
    parser.add_argument("--train_records", required=True, help="Path to training TFRecords")
    parser.add_argument("--student_type", default="listnet", choices=list(STUDENT_MAP.keys()))
    parser.add_argument("--output_path", default="./models/distilled_student.keras")
    parser.add_argument("--temperature", type=float, default=2.0, 
                        help="Soften the labels (T > 1.0 reveals more feature relationships)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()
    distill(args)